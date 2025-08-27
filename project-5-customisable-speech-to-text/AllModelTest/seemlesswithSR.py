import os
import logging
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import torch
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model
from concurrent.futures import ThreadPoolExecutor
import fasttext
from huggingface_hub import hf_hub_download
import io
import speech_recognition as sr
from queue import Queue

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
    def __init__(self, sample_rate=16000, chunk_size=1024, energy_threshold=1000, record_timeout=2):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.energy_threshold = energy_threshold
        self.record_timeout = record_timeout
        self.temp_files = []
        self.audio_stream = None
        self.transcriber = SpeechToTextTranscriber()
        self.data_queue = Queue()
        self.last_sample = bytes()

        # Initialize Silero VAD model
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                               model='silero_vad',
                                               force_reload=False)
        self.get_speech_timestamps, _, self.read_audio, _, _ = utils

    def start_stream(self):
        """Starts the audio stream with real-time processing."""
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.energy_threshold
        self.recorder.dynamic_energy_threshold = False

        with sr.Microphone(sample_rate=self.sample_rate) as source:
            self.recorder.adjust_for_ambient_noise(source)

        self.audio_stream = self.recorder.listen_in_background(sr.Microphone(sample_rate=self.sample_rate), self.audio_callback, phrase_time_limit=self.record_timeout)
        logging.info("Recording started... Speak into your microphone.")

    def audio_callback(self, _, audio: sr.AudioData) -> None:
        """Callback function to receive audio data when recordings finish."""
        data = audio.get_raw_data()
        self.data_queue.put(data)

    def process_audio(self):
        """Processes the audio input and detects speech."""
        while True:
            if not self.data_queue.empty():
                while not self.data_queue.empty():
                    data = self.data_queue.get()
                    self.last_sample += data

                audio_data = sr.AudioData(self.last_sample, self.sample_rate, 2)  # Assuming 16-bit audio
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Save temporary WAV file
                temp_file = f"temp_audio_{len(self.temp_files)}.wav"
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())
                self.temp_files.append(temp_file)

                # Detect speech using Silero VAD
                wav = self.read_audio(temp_file, sampling_rate=self.sample_rate)
                speech_timestamps = self.get_speech_timestamps(wav, self.vad_model, sampling_rate=self.sample_rate)

                if speech_timestamps:
                    logging.info('Speech Detected!')
                    self.transcriber.transcribe(temp_file)
                else:
                    logging.info('Silence Detected!')

                self.last_sample = bytes()

    def stop_stream(self):
        """Stops the audio stream and ensures all transcriptions complete."""
        if self.audio_stream:
            self.audio_stream(wait_for_stop=True)
            logging.info("Recording stopped.")
            self.transcriber.wait_for_completion()
            self.delete_temp_files()

    def delete_temp_files(self):
        """Deletes all temporary audio files."""
        for file in self.temp_files:
            os.remove(file)
            logging.info(f"Deleted {file}.")
        self.temp_files.clear()

class SpeechToTextTranscriber:
    def __init__(self):
        self.device = torch.device("cpu")
        model_id = "facebook/seamless-m4t-v2-large"

        self.model = SeamlessM4Tv2Model.from_pretrained(model_id).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.fasttext_model = self.load_fasttext_model()
        self.executor = ThreadPoolExecutor(max_workers=1)

    def load_fasttext_model(self):
        """Load the FastText language identification model."""
        model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        return fasttext.load_model(model_path)

    def transcribe(self, file_path):
        """Transcribes the given audio file and verifies the language in a separate thread."""
        self.executor.submit(self._transcribe_file, file_path)

    def _transcribe_file(self, file_path):
        try:
            logging.info(f"Transcribing {file_path}...")

            waveform, sample_rate = self.load_audio(file_path)
            
            inputs = self.processor(audios=waveform.numpy(), return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(**inputs, tgt_lang="eng", return_timestamps=True)

            transcription = self.processor.decode(outputs[0].cpu(), skip_special_tokens=True)

            language = self.identify_language(transcription)
            logging.info(f"Language detected: {language}")
            logging.info(f"Transcription: {transcription}")
        except Exception as e:
            logging.error(f"Error transcribing {file_path}: {str(e)}")

    def load_audio(self, file_path):
        """Load and preprocess audio file."""
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform, 16000

    def identify_language(self, text):
        """Identifies the language of the given text using FastText."""
        predictions = self.fasttext_model.predict(text, k=1)
        return predictions[0][0]

    def wait_for_completion(self):
        """Waits for all transcription threads to complete."""
        self.executor.shutdown(wait=True)

def main():
    audio_capture = AudioCapture()
    audio_capture.start_stream()

    try:
        while True:
            audio_capture.process_audio()
    except KeyboardInterrupt:
        print("\nStopping the recording...")
    finally:
        audio_capture.stop_stream()

if __name__ == "__main__":
    main()