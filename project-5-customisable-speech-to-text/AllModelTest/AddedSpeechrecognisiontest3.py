import io
import os
import logging
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from concurrent.futures import ThreadPoolExecutor
import fasttext
from huggingface_hub import hf_hub_download
from queue import Queue
from datetime import datetime
from tempfile import NamedTemporaryFile
import speech_recognition as sr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("transcription.log"),
        logging.StreamHandler()
    ]
)

class AudioCapture:
    def __init__(self, sample_rate=16000, chunk_size=1024, silence_threshold=0.01, silence_duration=0.5):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.temp_files = []
        self.audio_stream = None
        self.transcriber = SpeechToTextTranscriber()
        self.audio_processor = AudioProcessor(sample_rate, chunk_size, silence_duration, silence_threshold)

    def start_stream(self):
        self.audio_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.chunk_size,
            callback=self.audio_processor.process_audio
        )
        self.audio_stream.start()
        logging.info("Recording started... Speak into your microphone.")
    
    def stop_stream(self):
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            logging.info("Recording stopped.")
            self.finish_transcriptions()

    def finish_transcriptions(self):
        logging.info("Waiting for all transcriptions to complete...")
        self.transcriber.wait_for_completion()
        logging.info("All transcriptions are complete.")
        self.play_or_delete_options()

    def save_and_transcribe_audio(self, audio_data, segment_number):
        file_name = f"temp_audio_segment_{segment_number}.wav"
        audio_data_int16 = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
        write(file_name, self.sample_rate, audio_data_int16)
        self.temp_files.append(file_name)
        logging.info(f"Audio segment saved as {file_name}.")
        self.transcriber.transcribe(file_name)

    def play_or_delete_options(self):
        while True:
            action = input("Press 'p' to play all saved audio files, 'd' to delete them, or 'q' to quit: ").lower()
            if action == 'p':
                self.play_files()
            elif action == 'd':
                self.delete_files()
            elif action == 'q':
                logging.info("Exiting...")
                break
            else:
                logging.warning("Invalid input, please try again.")

    def play_files(self):
        for file in self.temp_files:
            logging.info(f"Playing {file}...")
            os.system(f"ffplay -nodisp -autoexit {file}")

    def delete_files(self):
        for file in self.temp_files:
            os.remove(file)
            logging.info(f"Deleted {file}.")
        self.temp_files.clear()

class AudioProcessor:
    def __init__(self, sample_rate=16000, chunk_size=1024, silence_duration=0.5, silence_threshold=0.01):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_duration = silence_duration
        self.silence_threshold = silence_threshold
        self.segment_number = 0
        self.current_audio = np.array([], dtype=np.float32)
        self.silence_count = 0
        self.MIN_AUDIO_LENGTH = int(self.sample_rate * 0.5)
        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        self.get_speech_timestamps, _, self.read_audio, _, _ = self.utils
        self.temp_file = NamedTemporaryFile(delete=False, suffix=".wav").name
        self.data_queue = Queue()
        self.last_sample = bytes()
        
    def process_audio(self, indata, frames, time, status):
        if status:
            logging.warning(f"Stream status: {status}")
        audio_data = np.frombuffer(indata, dtype=np.float32)
        logging.debug(f"Received audio chunk with max amplitude: {np.max(np.abs(audio_data))}")

        if np.max(np.abs(audio_data)) < self.silence_threshold:
            self.silence_count += 1
            if self.silence_count > (self.silence_duration * self.sample_rate / self.chunk_size):
                logging.info("Silence Detected.")
            return

        self.silence_count = 0
        self.current_audio = np.concatenate((self.current_audio, audio_data))
        logging.debug(f"Current audio length: {len(self.current_audio)} samples.")


        audio_data_bytes = indata.tobytes()
        self.data_queue.put(audio_data_bytes)

        if not self.data_queue.empty():
            while not self.data_queue.empty():
                self.last_sample += self.data_queue.get()

            write(self.temp_file, self.sample_rate, np.frombuffer(self.last_sample, dtype=np.float32))
            logging.info(f"Saved temporary audio to {self.temp_file} for speech detection.")


            wav = self.read_audio(self.temp_file)
            speech_timestamps = self.get_speech_timestamps(wav, self.model, sampling_rate=self.sample_rate)

            if speech_timestamps:
                logging.info(f"Speech Detected: {speech_timestamps}")
                audio_capture.save_and_transcribe_audio(self.current_audio, self.segment_number)
                self.segment_number += 1
                self.current_audio = np.array([], dtype=np.float32)

        self.last_sample = bytes()


class SpeechToTextTranscriber:
    def __init__(self):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "mps" else torch.float32

        model_id = "openai/whisper-large-v3"

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="sdpa"
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device
        )

        self.fasttext_model = self.load_fasttext_model()
        self.executor = ThreadPoolExecutor(max_workers=1)

    def load_fasttext_model(self):
        model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        return fasttext.load_model(model_path)

    def transcribe(self, file_path):
        self.executor.submit(self._transcribe_file, file_path)

    def _transcribe_file(self, file_path):
        try:
            logging.info(f"Transcribing {file_path}...")
            generate_kwargs = {
                "return_timestamps": True,
            }

            result = self.pipe(file_path, generate_kwargs=generate_kwargs)
            text = result['text']

            language = self.identify_language(text)
            logging.info(f"Language detected: {language}")
            logging.info(f"Transcription for {file_path}: {text}")
        except Exception as e:
            logging.error(f"Error transcribing {file_path}: {e}")

    def identify_language(self, text):
        predictions = self.fasttext_model.predict(text, k=1)
        return predictions[0][0]

    def wait_for_completion(self):
        self.executor.shutdown(wait=True)

    def warm_up(self):
        dummy_audio = np.zeros((16000,))
        temp_file = "warm_up_audio.wav"
        write(temp_file, 16000, dummy_audio.astype(np.int16))
        logging.info("Performing warm-up transcription...")
        self._transcribe_file(temp_file)
        os.remove(temp_file)
        logging.info("Warm-up complete.")

if __name__ == "__main__":
    sample_rate = 16000
    chunk_size = 1024
    silence_duration = 1
    silence_threshold = 0.01

    audio_capture = AudioCapture(sample_rate=sample_rate, chunk_size=chunk_size, silence_threshold=silence_threshold, silence_duration=silence_duration)

    audio_capture.transcriber.warm_up()

    try:
        audio_capture.start_stream()
        input("Press Enter to stop recording...\n")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        audio_capture.stop_stream()