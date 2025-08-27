import os
import logging
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import transformers
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import torch
from concurrent.futures import ThreadPoolExecutor
from speechbrain.inference.classifiers import EncoderClassifier
from unidecode import unidecode
import torchaudio
import fasttext
from huggingface_hub import hf_hub_download
import torch.nn.functional as F  # Import softmax for probability calculations


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
        self.audio_processor = AudioProcessor(self.transcriber, self)

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
        try:
            file_name = f"temp_audio_segment_{segment_number}.wav"
            if np.max(np.abs(audio_data)) == 0:
                logging.warning(f"Skipping empty audio segment {segment_number}")
                return None
            audio_data_int16 = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
            write(file_name, self.sample_rate, audio_data_int16)
            self.temp_files.append(file_name)
            logging.info(f"Audio segment saved as {file_name}.")
            return file_name
        except Exception as e:
            logging.error(f"Error saving audio segment: {e}")
            return None

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
    def __init__(self, transcriber, audio_capture, sample_rate=16000, chunk_size=1024, silence_duration=1, silence_threshold=0.01):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_duration = silence_duration
        self.silence_threshold = silence_threshold
        self.segment_number = 0
        self.current_audio = np.array([], dtype=np.float32)
        self.silence_count = 0
        self.MIN_AUDIO_LENGTH = self.sample_rate * 0.5  # 0.5 seconds
        self.transcriber = transcriber
        self.audio_capture = audio_capture

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
                file_name = self.audio_capture.save_and_transcribe_audio(self.current_audio, self.segment_number)
                if file_name:
                    self.transcriber.transcribe(file_name) 
                self.segment_number += 1
                self.current_audio = np.array([], dtype=np.float32)
            self.silence_count = 0

class SpeechToTextTranscriber:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Load Whisper model for transcription
        model_id = "openai/whisper-small"
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, 
            use_safetensors=True, attn_implementation="sdpa").to(self.device)
        self.processor = WhisperProcessor.from_pretrained(model_id)

        # Load MMS model for language identification (on CPU to prevent OOM issues)
        self.mms_lid_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "facebook/mms-lid-126", torch_dtype=torch_dtype, low_cpu_mem_usage=True, 
            use_safetensors=True).to(self.device) 
        self.mms_lid_processor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-126")

        # Load FastText model for text-based language identification
        self.fasttext_model = self.load_fasttext_model()

        # Load SpeechBrain VoxLingua for audio language identification
        self.audio_language_model = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")

        self.executor = ThreadPoolExecutor(max_workers=3)  # Increase workers for concurrent transcriptions

    def load_fasttext_model(self):
        """Loads the FastText model for language identification."""
        fasttext_model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        return fasttext.load_model(fasttext_model_path)

    def transcribe_real_time(self, audio_input):
        """Performs real-time transcription on the provided audio data."""
        try:
            # Detect language from the audio
            detected_language_list = self.detect_language_from_audio(audio_input)

            # Prepare audio data for Whisper
            input_features = self.processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)

            # Generate token ids
            predicted_ids = self.model.generate(input_features)

            # Decode the predicted IDs into text
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            logging.info(f"Real-time transcription: {transcription}")

            if detected_language_list:
                logging.info(f"Direct audio language detection: {detected_language_list}")
                # Ensure transcription is in English letters (Romanized form)
                if detected_language_list[0] != "en":
                    transcription = self.romanize_text(transcription, detected_language_list[0])
                    logging.info(f"Transcription in English letters: {transcription}")

            # Detect language from the text using FastText
            detected_text_language = self.identify_language(transcription)
            logging.info(f"Text-based language detection (FastText): {detected_text_language}")

        except RuntimeError as re:
            logging.error(f"CUDA error during transcription: {re}")
        except transformers.exceptions.GenerationError as ge:
            logging.error(f"Whisper generation error: {ge}") 
        except Exception as e:
            logging.error(f"Error during transcription: {e}")

    def romanize_text(self, text, detected_language):
        """Romanizes the text if the detected language is non-Latin."""
        try:
            romanized_text = unidecode(text)
            logging.info(f"Romanized text: {romanized_text}")
            return romanized_text
        except Exception as e:
            logging.error(f"Error romanizing text: {e}")
            return text

    def detect_language_from_audio(self, audio_input):
        """Detects language from the provided audio input using MMS LID."""
        try:
            audio_features = self.mms_lid_processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values
            with torch.no_grad():
                logits = self.mms_lid_model(audio_features.to(self.device)).logits
                probabilities = F.softmax(logits, dim=-1)
                predicted_label_id = torch.argmax(probabilities, dim=-1).item()
                return [self.mms_lid_model.config.id2label[predicted_label_id]]
        except Exception as e:
            logging.error(f"Error detecting language from audio: {e}")
            return None

    def identify_language(self, text):
        """Identifies the language of the provided text using FastText."""
        try:
            text = text.lower()
            detected_language = self.fasttext_model.predict(text)[0][0].replace("__label__", "")
            logging.info(f"Detected text language: {detected_language}")
            return detected_language
        except Exception as e:
            logging.error(f"Error identifying text language: {e}")
            return None

    def transcribe(self, audio_file):
        """Starts the transcription of the provided audio file in a separate thread."""
        logging.info(f"Starting transcription for {audio_file}...")
        self.executor.submit(self.transcribe_audio_file, audio_file)

    def transcribe_audio_file(self, audio_file):
        """Transcribes the given audio file and logs the result."""
        try:
            waveform, sample_rate = torchaudio.load(audio_file)
            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
            audio_input = waveform.squeeze(0).numpy()

            # Perform real-time transcription
            self.transcribe_real_time(audio_input)

        except Exception as e:
            logging.error(f"Error during file transcription: {e}")

    def wait_for_completion(self):
        """Waits for all transcription tasks to complete."""
        self.executor.shutdown(wait=True)

if __name__ == "__main__":
    audio_capture = AudioCapture()
    try:
        audio_capture.start_stream(audio_capture.audio_processor.process_audio)
        while True:
            command = input("Press 's' to stop the recording: ").lower()
            if command == 's':
                audio_capture.stop_stream()
                break
    except KeyboardInterrupt:
        audio_capture.stop_stream()