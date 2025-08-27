import os
import logging
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from concurrent.futures import ThreadPoolExecutor
import fasttext
from huggingface_hub import hf_hub_download
from speechbrain.inference.classifiers import EncoderClassifier
from unidecode import unidecode
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import soundfile as sf

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
    def __init__(self, sample_rate=16000, chunk_size=1024, silence_threshold=0.05, silence_duration=2):
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
    def __init__(self, sample_rate=16000, chunk_size=1024, silence_duration=2, silence_threshold=0.05):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_duration = silence_duration
        self.silence_threshold = silence_threshold
        self.segment_number = 0
        self.current_audio = np.array([], dtype=np.float32)
        self.silence_count = 0
        self.MIN_AUDIO_LENGTH = self.sample_rate * 1  # Minimum 0.5 seconds of audio

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

        # If silence persists for the specified duration, check if audio should be saved
        if self.silence_count >= (self.silence_duration * self.sample_rate / self.chunk_size):
            # Save only if the length of the audio is longer than the minimum audio length
            if len(self.current_audio) > self.MIN_AUDIO_LENGTH:
                audio_capture.save_and_transcribe_audio(self.current_audio, self.segment_number)
                self.segment_number += 1
            else:
                logging.info("Discarding short or silent audio segment.")
            
            # Reset the current audio and silence counter after saving or discarding
            self.current_audio = np.array([], dtype=np.float32)
            self.silence_count = 0

class SpeechToTextTranscriber:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the Wav2Vec2 model and processor
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(device)
        self.device = device

        # Load the VoxLingua107 model for language detection
        self.audio_language_model = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")

        # Load FastText model for text-based language identification
        self.fasttext_model = self.load_fasttext_model()

        self.executor = ThreadPoolExecutor(max_workers=1)

    def load_fasttext_model(self):
        """Downloads and loads the FastText model for text-based language identification."""
        fasttext_model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        return fasttext.load_model(fasttext_model_path)

    def transcribe(self, file_path):
        """Submits the transcription task to the ThreadPoolExecutor."""
        self.executor.submit(self._transcribe, file_path)

    def _transcribe(self, file_path):
        """Performs the transcription of the provided audio file."""
        try:
            # Step 1: Detect the language from the audio using VoxLingua107
            detected_language_list = self.detect_language_from_audio(file_path)
            detected_language_code = detected_language_list[0].split(":")[0] if detected_language_list else None

            if detected_language_code:
                logging.info(f"Detected language code from audio: {detected_language_code}")

            # Step 2: Perform transcription using Wav2Vec2
            transcription = self.transcribe_audio_with_wav2vec2(file_path)
            logging.info(f"Transcription for {file_path}: {transcription}")

            # Step 3: Detect the language of the transcribed text using FastText
            language_from_text = self.identify_language(transcription)
            logging.info(f"Detected language from transcribed text: {language_from_text}")

            # Step 4: If the detected language is not English, attempt to romanize the text
            if detected_language_code and detected_language_code != "en":
                transcription = self.romanize_text(transcription, detected_language_code)
                logging.info(f"Romanized transcription: {transcription}")

        except Exception as e:
            logging.error(f"Error during transcription: {e}")

    def transcribe_audio_with_wav2vec2(self, file_path):
        """Transcribes the audio file using Wav2Vec2."""
        # Load audio file
        audio_input, sample_rate = sf.read(file_path)

        # Preprocess audio
        input_values = self.processor(audio_input, return_tensors="pt", sampling_rate=sample_rate).input_values.to(self.device)

        # Perform transcription
        with torch.no_grad():
            logits = self.model(input_values).logits

        # Decode the predicted tokens to text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])

        return transcription

    def detect_language_from_audio(self, file_path):
        """Detects the language of the given audio file using VoxLingua107."""
        try:
            # Load the audio file
            signal = self.audio_language_model.load_audio(file_path)
            # Classify the language from the audio signal
            prediction = self.audio_language_model.classify_batch(signal)
            # Extract the predicted language ISO code
            audio_language = prediction[3]  # This contains the ISO code of the detected language
            
            logging.info(f"Detected language from audio: {audio_language}")
            return audio_language
        except Exception as e:
            logging.error(f"Error detecting language from audio: {e}")
            return None

    def romanize_text(self, text, language_code):
        """Converts text to English letters (Romanized form) if necessary."""
        # Use unidecode for basic romanization, can be replaced by specific libraries for more languages
        if language_code != 'en':
            return unidecode(text)
        else:
            return text  

    def identify_language(self, text):
        """Identifies the language of the given text using FastText."""
        predictions = self.fasttext_model.predict(text, k=1)
        return predictions[0][0]

    def wait_for_completion(self):
        """Waits for all transcriptions to finish."""
        self.executor.shutdown(wait=True)

# Main execution flow stays the same
if __name__ == "__main__":
    silence_duration = 2

    # Initialize audio capture
    audio_capture = AudioCapture(silence_duration=silence_duration)
    audio_processor = AudioProcessor(silence_duration=silence_duration)

    try:
        # Start recording and processing
        audio_capture.start_stream(audio_processor.process_audio)
        input("Press Enter to stop recording...\n")  # User stops recording manually
    finally:
        # Ensure the stream is stopped and processed properly
        audio_capture.stop_stream()
