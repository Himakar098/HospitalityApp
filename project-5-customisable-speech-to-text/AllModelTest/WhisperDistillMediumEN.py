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
    def __init__(self, sample_rate=16000, chunk_size=1024, silence_threshold=0.01, silence_duration=2):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.temp_files = []  # List to store temporary file paths
        self.audio_stream = None
        self.transcriber = SpeechToTextTranscriber()  # Instantiate transcriber

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
        """Provides options to play or delete the saved audio files."""
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

class AudioProcessor:
    def __init__(self, sample_rate=16000, chunk_size=1024, silence_duration=2, silence_threshold=0.01):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_duration = silence_duration
        self.silence_threshold = silence_threshold
        self.segment_number = 0
        self.current_audio = np.array([], dtype=np.float32)
        self.silence_count = 0
        self.MIN_AUDIO_LENGTH = self.sample_rate * 0.5  # 0.5 seconds

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

class SpeechToTextTranscriber:
    def __init__(self):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        torch_dtype = torch.float32

        model_id = "distil-whisper/distil-medium.en"

        # Load the Whisper model and processor
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="eager"
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Create a speech recognition pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device
        )

        self.fasttext_model = self.load_fasttext_model()
        self.executor = ThreadPoolExecutor(max_workers=1)  # List to keep track of transcription threads

    def load_fasttext_model(self):
        """Load the FastText language identification model."""
        model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        return fasttext.load_model(model_path)

    def transcribe(self, file_path):
        """Transcribes the given audio file and verifies the language in a separate thread."""
        self.executor.submit(self._transcribe_file, file_path)

    def _transcribe_file(self, file_path):
        """Internal method to handle transcription and language identification."""
        try:
            logging.info(f"Transcribing {file_path}...")
            result = self.pipe(file_path, return_timestamps=True)
            text = result['text']

            # Perform language identification using FastText
            language = self.identify_language(text)
            logging.info(f"Language detected: {language}")
            logging.info(f"Transcription for {file_path}: {text}")
        except Exception as e:
            logging.error(f"Error transcribing {file_path}: {e}")

    def identify_language(self, text):
        """Identifies the language of the given text using FastText."""
        predictions = self.fasttext_model.predict(text, k=1)
        return predictions[0][0]

    def wait_for_completion(self):
        """Waits for all transcription threads to complete."""
        self.executor.shutdown(wait=True)

    def warm_up(self):
        """Performs a warm-up transcription to speed up initial response times."""
        dummy_audio = np.zeros((16000,))  # 1 second of silence at 16kHz
        temp_file = "warm_up_audio.wav"
        write(temp_file, 16000, dummy_audio.astype(np.int16))
        logging.info("Performing warm-up transcription...")
        self._transcribe_file(temp_file)  # Warm-up with dummy file
        os.remove(temp_file)
        logging.info("Warm-up complete.")

# Main execution
if __name__ == "__main__":
    sample_rate = 16000
    chunk_size = 1024
    silence_duration = 2
    silence_threshold = 0.01

    audio_processor = AudioProcessor(sample_rate=sample_rate, chunk_size=chunk_size, silence_duration=silence_duration, silence_threshold=silence_threshold)
    audio_capture = AudioCapture(sample_rate=sample_rate, chunk_size=chunk_size, silence_threshold=silence_threshold, silence_duration=silence_duration)

    # Perform a warm-up to reduce initial latency
    audio_capture.transcriber.warm_up()

    try:
        audio_capture.start_stream(audio_processor.process_audio)
        input("Press Enter to stop recording...\n")  # Keep running until user stops
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        audio_capture.stop_stream()