import os
import logging
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
import torch
from concurrent.futures import ThreadPoolExecutor
from speechbrain.pretrained import EncoderClassifier
from unidecode import unidecode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("transcription.log"), logging.StreamHandler()]
)

class AudioCapture:
    def __init__(self, sample_rate=16000, chunk_size=1024, silence_threshold=0.01, silence_duration=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.temp_files = []  # Store temp file paths
        self.audio_stream = None
        self.transcriber = SpeechToTextTranscriber()

    def start_stream(self, callback):
        self.audio_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.chunk_size,
            callback=callback
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
        write(file_name, self.sample_rate, audio_data_int16)  # Save as 16-bit PCM
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
    def __init__(self, sample_rate=16000, chunk_size=1024, silence_duration=1, silence_threshold=0.01):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_duration = silence_duration
        self.silence_threshold = silence_threshold
        self.segment_number = 0
        self.current_audio = np.array([], dtype=np.float32)
        self.silence_count = 0
        self.MIN_AUDIO_LENGTH = self.sample_rate * 0.5  # Minimum 0.5 seconds

    def process_audio(self, indata, frames, time, status):
        if status:
            logging.warning(f"Stream status: {status}")
        audio_data = np.frombuffer(indata, dtype=np.float32)
        self.current_audio = np.concatenate((self.current_audio, audio_data))

        # RMS-based silence detection
        rms = np.sqrt(np.mean(audio_data ** 2))

        if rms < self.silence_threshold:
            self.silence_count += 1
        else:
            self.silence_count = 0

        # Save the audio segment if enough silence has been detected
        if self.silence_count >= (self.silence_duration * self.sample_rate / self.chunk_size):
            if len(self.current_audio) > self.MIN_AUDIO_LENGTH:
                audio_capture.save_and_transcribe_audio(self.current_audio, self.segment_number)
                self.segment_number += 1
                self.current_audio = np.array([], dtype=np.float32)
            self.silence_count = 0

class SpeechToTextTranscriber:
    def __init__(self):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "mps" else torch.float32
        model_id = "openai/whisper-small"

        # Load Whisper model and processor
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(device)
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.pipe = pipeline("automatic-speech-recognition", model=self.model, tokenizer=self.processor.tokenizer,
                             feature_extractor=self.processor.feature_extractor, device=device)

        # Load the LID model for language detection
        self.audio_language_model = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.language_mapping = {
            "en": "english", "zh": "chinese", "de": "german", "es": "spanish", "fr": "french",
            "it": "italian", "ko": "korean", "ja": "japanese", "pt": "portuguese", "tr": "turkish"
        }

    def transcribe(self, file_path):
        self.executor.submit(self._transcribe, file_path)

    def _transcribe(self, file_path):
        try:
            detected_language = self.detect_language_from_audio(file_path)
            whisper_language = self.language_mapping.get(detected_language, None)
            if whisper_language:
                logging.info(f"Language detected: {whisper_language}. Using Whisper for transcription.")
                result = self.pipe(file_path, generate_kwargs={"language": whisper_language})
            else:
                logging.warning(f"Language detection failed, defaulting to Whisper's auto language detection.")
                result = self.pipe(file_path)
            transcription = result['text']
            logging.info(f"Transcription for {file_path}: {transcription}")

        except Exception as e:
            logging.error(f"Error during transcription: {e}")

    def detect_language_from_audio(self, file_path):
        try:
            signal = self.audio_language_model.load_audio(file_path)
            prediction = self.audio_language_model.classify_batch(signal)
            audio_language = prediction[3]
            logging.info(f"Detected language from audio: {audio_language}")
            return audio_language
        except Exception as e:
            logging.error(f"Error detecting language: {e}")
            return None

    def wait_for_completion(self):
        self.executor.shutdown(wait=True)

# Main execution
if __name__ == "__main__":
    silence_duration = 2
    audio_capture = AudioCapture(silence_duration=silence_duration)
    audio_processor = AudioProcessor()

    try:
        # Start recording and processing
        audio_capture.start_stream(audio_processor.process_audio)
        input("Press Enter to stop recording...\n")
    finally:
        audio_capture.stop_stream()
