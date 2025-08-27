import os
import logging
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from transformers import Wav2Vec2ForCTC, AutoProcessor, Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import torch
import torchaudio
from concurrent.futures import ThreadPoolExecutor

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
        self.play_or_delete_options()

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
    def __init__(self, audio_capture, sample_rate=16000, chunk_size=1024, silence_duration=1, silence_threshold=0.01):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_duration = silence_duration
        self.silence_threshold = silence_threshold
        self.segment_number = 0
        self.current_audio = np.array([], dtype=np.float32)
        self.silence_count = 0
        self.MIN_AUDIO_LENGTH = self.sample_rate * 0.5  # Minimum of 0.5 seconds of audio for processing
        self.audio_capture = audio_capture  # Reference to AudioCapture instance

    def process_audio(self, indata, frames, time, status):
        """Processes the audio input from the stream callback."""
        if status:
            logging.warning(f"Stream status: {status}")
        
        # Convert incoming audio data to NumPy array
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
                # Ensure the audio is long enough
                if len(self.current_audio) >= self.MIN_AUDIO_LENGTH:
                    self.audio_capture.save_and_transcribe_audio(self.current_audio, self.segment_number)
                    self.segment_number += 1
                    self.current_audio = np.array([], dtype=np.float32)  # Clear after saving
            self.silence_count = 0

class SpeechToTextTranscriber:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        #self.torch_dtype = torch.float16 if self.device == "mps" else torch.float32

        self.model_id = "facebook/mms-1b-all"

        # Load the Wav2Vec2 model and processor for transcription
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # Load the MMS model for language identification
        self.model_lid = "facebook/mms-lid-2048"
        self.lid_model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_lid).to(self.device)  #"facebook/mms-lid-4017"
        self.lid_processor = AutoFeatureExtractor.from_pretrained(self.model_lid)

        self.executor = ThreadPoolExecutor(max_workers=2)

    def transcribe(self, file_path):
        """Submits the transcription task to the ThreadPoolExecutor."""
        self.executor.submit(self._transcribe, file_path)

    def _transcribe(self, file_path):
        """Performs the transcription of the provided audio file."""
        try:
            # Detect the language using MMS LID
            detected_language = self.detect_language(file_path)
            if detected_language:
                logging.info(f"Detected language: {detected_language}")
            else:
                logging.warning("Could not detect language. Defaulting to English.")
                detected_language = "eng"
            
            # Load the audio data
            audio_input, sample_rate = torchaudio.load(file_path)
        
            # Resample if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                audio_input = resampler(audio_input)

            audio_input = audio_input.squeeze().to(self.device)   # Remove extra channel dimension if present, keep [batch_size, sequence_length]

            # Process the audio through the processor
            inputs = self.processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = inputs.to(self.device)
            # Get logits from the model
            with torch.no_grad():
                logits = self.model(**inputs).logits
        
            # Decode the logits to get the transcription
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription_text = self.processor.batch_decode(predicted_ids)[0]
        
            logging.info(f"Transcription for {file_path}: {transcription_text}")
        
        except Exception as e:
            logging.error(f"Error in transcription: {e}")

    def detect_language(self, file_path):
        """Detects the language of the audio file using the MMS LID model."""
        try:
            audio_input, sample_rate = torchaudio.load(file_path)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                audio_input = resampler(audio_input)
            audio_input = audio_input.squeeze().to(self.device)  # Move input to MPS

            # Preprocess the input for language identification
            inputs = self.lid_processor(audio_input, return_tensors="pt", sampling_rate=16000)
            inputs = inputs.to(self.device)  # Move inputs to MPS

            with torch.no_grad():
                logits = self.lid_model(**inputs).logits

            predicted_language_id = torch.argmax(logits, dim=-1).item()

            # Decode the logits to get the language label
            detected_language = self.lid_model.config.id2label[predicted_language_id]
            return detected_language
        
        except Exception as e:
            logging.error(f"Error in language detection: {e}")
            return None

    def wait_for_completion(self):
        """Waits for all tasks to complete."""
        self.executor.shutdown(wait=True)

if __name__ == "__main__":
    # Initialize the audio capture and processing components
    audio_capture = AudioCapture()
    audio_processor = AudioProcessor(audio_capture)

    try:
        audio_capture.start_stream(audio_processor.process_audio)
        input("Press Enter to stop the recording.\n")
    finally:
        audio_capture.stop_stream()
