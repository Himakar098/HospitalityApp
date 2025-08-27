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
from speechbrain.inference.classifiers import EncoderClassifier

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
    def __init__(self, sample_rate=16000, chunk_size=1024, silence_threshold=0.01, silence_duration=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.temp_files = []  # List to store temporary file paths
        self.audio_stream = None
        self.transcriber = SpeechToTextTranscriber()  # Instantiate transcriber
        self.segment_number = 0
        self.current_audio = np.array([], dtype=np.float32)

    def start_stream(self, callback):
        """Starts the audio stream with the given callback for processing."""
        self.audio_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.chunk_size,
            callback=callback
        )
        self.audio_stream.start()
        logging.info("Recording started... Speak into your microphone.")
    
    def stop_stream(self):
        """Stops the audio stream and ensures all transcriptions complete."""
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            logging.info("Recording stopped.")
            self.finish_transcriptions()

    def finish_transcriptions(self):
        """Waits for all transcription threads to complete."""
        logging.info("Waiting for all transcriptions to complete...")
        self.transcriber.wait_for_completion()
        logging.info("All transcriptions are complete.")
        self.play_or_delete_options()  # After transcription, offer options to the user

    def save_and_transcribe_audio(self, audio_data, segment_number):
        """Saves the audio data to a temporary file and starts transcription in a separate thread."""
        file_name = f"temp_audio_segment_{segment_number}.wav"
        
        # Normalize audio_data to int16 range (-32768 to 32767)
        audio_data_int16 = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
        
        write(file_name, self.sample_rate, audio_data_int16)  # Save as 16-bit PCM
        self.temp_files.append(file_name)
        logging.info(f"Audio segment saved as {file_name}.")
        
        # Start a new thread to handle transcription
        self.transcriber.transcribe(file_name)

    def play_or_delete_options(self):
        """Provides options to play, delete, or re-transcribe saved audio files."""
        while True:
            action = input("Press 'p' to play all saved audio files, 'd' to delete them, 'r' to re-transcribe, or 'q' to quit: ").lower()
            if action == 'p':
                self.play_files()
            elif action == 'd':
                self.delete_files()
            elif action == 'r':
                self.retranscribe_files()
            elif action == 'q':
                logging.info("Exiting...")
                break
            else:
                logging.warning("Invalid input, please try again.")

    def play_files(self):
        """Plays all saved audio files."""
        for file in self.temp_files:
            logging.info(f"Playing {file}...")
            os.system(f"ffplay -nodisp -autoexit {file}")

    def delete_files(self):
        """Deletes all saved audio files."""
        for file in self.temp_files:
            os.remove(file)
            logging.info(f"Deleted {file}.")
        self.temp_files.clear()

    def retranscribe_files(self):
        """Re-transcribes all saved audio files."""
        for file in self.temp_files:
            self.transcriber.transcribe(file)


class AudioProcessor:
    def __init__(self, sample_rate=16000, chunk_size=1024, silence_duration=1, silence_threshold=0.01):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_duration = silence_duration
        self.silence_threshold = silence_threshold
        self.segment_number = 0
        self.current_audio = np.array([], dtype=np.float32)
        self.silence_count = 0
        self.MIN_AUDIO_LENGTH = self.sample_rate * 0.5  # 0.5 seconds
        self.MAX_AUDIO_LENGTH = self.sample_rate * 30  # 30 seconds
        self.SLIDING_WINDOW = self.sample_rate * 2  # 2 seconds

    def process_audio(self, indata, frames, time, status):
        """Processes the audio input from the stream callback."""
        if status:
            logging.warning(f"Stream status: {status}")
        audio_data = np.frombuffer(indata, dtype=np.float32)
        self.current_audio = np.concatenate((self.current_audio, audio_data))

        # Calculate RMS to detect silence
        rms = np.sqrt(np.mean(audio_data**2))

        if rms < self.silence_threshold:
            self.silence_count += 1
        else:
            self.silence_count = 0

        # If silence persists for the specified duration, save the audio segment
        if self.silence_count >= (self.silence_duration * self.sample_rate / self.chunk_size):
            if len(self.current_audio) > 0:
                audio_capture.save_and_transcribe_audio(self.current_audio, self.segment_number)
                self.segment_number += 1
                self.current_audio = np.array([], dtype=np.float32)  # Clear after saving
            self.silence_count = 0
        
        # If the audio exceeds 30 seconds, chunk it with a sliding window
        if len(self.current_audio) > self.MAX_AUDIO_LENGTH:
            logging.info("Chunking audio with sliding window due to exceeding length...")
            start = len(self.current_audio) - self.MAX_AUDIO_LENGTH
            end = len(self.current_audio) - self.SLIDING_WINDOW
            chunk = self.current_audio[start:end]
            audio_capture.save_and_transcribe_audio(chunk, self.segment_number)
            self.segment_number += 1
            self.current_audio = self.current_audio[-self.SLIDING_WINDOW:]  # Keep the last 2 seconds


class SpeechToTextTranscriber:
    def __init__(self):
        self.model_id = "distil-whisper/distil-large-v3"
        self.custom_model_path = "path/to/custom/model"  # Path to custom model
        self.model = None
        self.processor = None
        self.pipe = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.load_model(self.model_id)  # Load the default pretrained model

        self.audio_language_model = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")
        self.fasttext_model = self.load_fasttext_model()
        self.executor = ThreadPoolExecutor(max_workers=1)

    def load_model(self, model_path):
        """Loads either the pretrained model or the custom model."""
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=self.device
        )
        logging.info(f"Model {model_path} loaded.")

    def select_model(self):
        """Prompts the user to select the pretrained or custom model."""
        while True:
            choice = input("Press 'p' for pretrained model or 'c' for custom model: ").lower()
            if choice == 'p':
                self.load_model(self.model_id)
                break
            elif choice == 'c':
                self.load_model(self.custom_model_path)
                break
            else:
                logging.warning("Invalid choice, please try again.")

    def load_fasttext_model(self):
        """Load the FastText language identification model."""
        model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        return fasttext.load_model(model_path)

    def transcribe(self, file_path):
        """Transcribes the given audio file and verifies the language in a separate thread."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            self.executor.submit(self._transcribe_file, file_path)

    def _transcribe_file(self, file_path):
        """Internal method to handle transcription and language identification."""
        try:
            logging.info(f"Transcribing {file_path}...")
            
            # Detect language from audio first
            audio_language = self.detect_language_from_audio(file_path)            
            # Transcribe with the detected language
            result = self.pipe(file_path, generate_kwargs={"language": "english"}, return_timestamps=True)
            text = result.get('text', '') if isinstance(result, dict) else ""
            text_language = self.identify_language(text)
            
            logging.info(f"Audio language detected: {audio_language}")
            logging.info(f"Text language detected: {text_language}")
            logging.info(f"Transcription for {file_path}: {text}")
        except Exception as e:
            logging.error(f"Error transcribing {file_path}: {e}")

    def detect_language_from_audio(self, file_path):
        """Detects the language of the given audio file directly."""
        try:
            signal = self.audio_language_model.load_audio(file_path)
            prediction = self.audio_language_model.classify_batch(signal)
            audio_language = prediction[3]  # This contains the ISO code of the detected language
            return audio_language
        except Exception as e:
           # logging.error(f"Error detecting language from audio: {e}")
            return "en"  # Default to English in case of error

    def identify_language(self, text):
        """Identifies the language of the given text using FastText."""
        predictions = self.fasttext_model.predict(text, k=1)
        return predictions[0][0].replace('__label__', '')

    def wait_for_completion(self):
        """Waits for all transcriptions to finish."""
        self.executor.shutdown(wait=True)

# Main execution
if __name__ == "__main__":
    # Initialize audio capture
    audio_capture = AudioCapture()
    audio_processor = AudioProcessor()

    try:
        # Start recording and processing
        audio_capture.start_stream(audio_processor.process_audio)
        input("Press Enter to stop recording...\n")  # User stops recording manually
    finally:
        # Ensure the stream is stopped and processed properly
        audio_capture.stop_stream()

