import os
import logging
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model
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
    def __init__(self, sample_rate=16000, chunk_size=1024, silence_threshold=0.01, silence_duration=0.5):
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
    def __init__(self, sample_rate=16000, chunk_size=1024, silence_duration=0.5, silence_threshold=0.01):
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = "facebook/seamless-m4t-v2-large"

        # Load the SeamlessM4Tv2 model and processor
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
        """Internal method to handle transcription and language identification."""
        try:
            logging.info(f"Transcribing {file_path}...")

            # Load the audio file
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Check if the sample rate matches the model's expected rate (16000)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            
            # Ensure waveform is mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Process the waveform
            inputs = self.processor(audios=waveform.numpy(), return_tensors="pt", sampling_rate=16000)
            
            # Move inputs to the correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate transcription
            with torch.no_grad():
                outputs = self.model.generate(**inputs, tgt_lang="eng")

            # Decode the output
            transcription = self.processor.batch_decode(outputs, skip_special_tokens=True)

            # The transcription is now a list, so we join it into a single string
            transcription = " ".join(transcription)

            language = self.identify_language(transcription)
            logging.info(f"Language detected: {language}")
            logging.info(f"Transcription for {file_path}: {transcription}")
        except Exception as e:
            logging.error(f"Error transcribing {file_path}: {str(e)}")

    def identify_language(self, text):
        """Identifies the language of the given text using FastText."""
        predictions = self.fasttext_model.predict(text, k=1)
        return predictions[0][0]

    def wait_for_completion(self):
        """Waits for all transcription threads to complete."""
        self.executor.shutdown(wait=True)

    def warm_up(self):
        """Performs a warm-up transcription to speed up initial response times."""
        dummy_audio = torch.zeros((1, 16000)).to(self.device)
        logging.info("Performing warm-up transcription...")
        with torch.no_grad():
            inputs = self.processor(audios=dummy_audio.cpu().numpy(), return_tensors="pt", sampling_rate=16000)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            self.model.generate(**inputs, tgt_lang="eng")
        logging.info("Warm-up complete.")

# Main execution
if __name__ == "__main__":
    sample_rate = 16000
    chunk_size = 1024
    silence_duration = 1
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






