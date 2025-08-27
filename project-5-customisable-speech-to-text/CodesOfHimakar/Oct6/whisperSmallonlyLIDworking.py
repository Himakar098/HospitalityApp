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
from torchaudio import io
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
        try:
            file_name = f"temp_audio_segment_{segment_number}.wav"
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
    def __init__(self, audio_capture, sample_rate=16000, chunk_size=1024, silence_duration=2, silence_threshold=0.01):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_duration = silence_duration
        self.silence_threshold = silence_threshold
        self.segment_number = 0
        self.current_audio = np.array([], dtype=np.float32)
        self.silence_count = 0
        self.MIN_AUDIO_LENGTH = self.sample_rate * 0.5  # 0.5 seconds
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
                    self.audio_capture.transcriber.transcribe(file_name) 
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

        self.executor = ThreadPoolExecutor(max_workers=1)  # Create a ThreadPoolExecutor

    def load_fasttext_model(self):
        """Loads the FastText model for language identification."""
        fasttext_model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        return fasttext.load_model(fasttext_model_path)

    def transcribe(self, file_name):
        """Wrapper to handle transcription for a given audio file."""
        self.executor.submit(self.transcribe_real_time, file_name)

    def transcribe_real_time(self, audio_data):
        """Performs real-time transcription on the provided audio data."""
        try:
            # Detect language from the audio
            detected_language_list = self.detect_language_from_audio(audio_data)

            # Prepare audio data for Whisper
            input_features = self.processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)

            # Generate token ids
            predicted_ids = self.model.generate(input_features)

            # Decode the predicted IDs into text
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            logging.info(f"Real-time transcription: {transcription}")

            if detected_language_list:
                logging.info(f"Direct audio language detection: {detected_language_list}")
                transcription = self.romanize_text(transcription, detected_language_list[0])

            # Detect language from the text using FastText
            detected_text_language = self.identify_language(transcription)
            logging.info(f"Text-based language detection (FastText): {detected_text_language}")

        except RuntimeError as re:
            logging.error(f"CUDA error during transcription: {re}")
        except transformers.exceptions.GenerationError as ge:
            logging.error(f"Whisper generation error: {ge}") 
        except Exception as e:
            logging.error(f"Error in real-time transcription: {e}")

    def detect_language_from_audio(self, audio_data):
        """Detects the language of the given audio data directly."""
        try:
            audio_input, sample_rate = torchaudio.load(audio_data)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                audio_input = resampler(audio_input)
            audio_input = audio_input.squeeze().to(self.device)

            # Classify the language using MMS LID
            inputs = self.mms_lid_processor(audio_input, return_tensors="pt", sampling_rate=16000)
            inputs = inputs.to(self.device)
            with torch.no_grad():
                logits = self.mms_lid_model(**inputs).logits
            predicted_language_id = torch.argmax(logits, dim=-1).item()
            mms_language = self.mms_lid_model.config.id2label[predicted_language_id]

            # Classify the language using VoxLingua FastText
            prediction = self.audio_language_model.classify_batch(audio_input)
            voxlingua_language = prediction[3] 

            logging.info(f"MMS LID detected language: {mms_language}")
            logging.info(f"VoxLingua detected language: {voxlingua_language}")
            
            return [mms_language, voxlingua_language]
        except torchaudio.exceptions.TorchaudioException as te:
            logging.error(f"Error loading audio with torchaudio: {te}")
            return None
        except Exception as e:
            logging.error(f"Error detecting language from audio: {e}")
            return None

    def romanize_text(self, transcription, detected_language):
        """Romanizes the text if necessary."""
        if detected_language != "en":
            romanized_transcription = unidecode(transcription)
            logging.info(f"Text romanized from {detected_language} script: {romanized_transcription}")
            return romanized_transcription
        return transcription

    def identify_language(self, text):
        """Identifies the language of the given text using FastText."""
        predictions = self.fasttext_model.predict(text, k=1)
        language = predictions[0][0]
        confidence = predictions[1][0]  # Confidence score for the prediction
        return language, confidence

    def wait_for_completion(self):
        """Waits for all transcription threads to complete."""
        self.executor.shutdown(wait=True)

# Main execution
if __name__ == "__main__":
    # Create an instance of AudioCapture
    audio_capture = AudioCapture()

    # Create an instance of AudioProcessor and pass audio_capture to it
    audio_processor = AudioProcessor(audio_capture=audio_capture)

    try:
        # Start the stream with the audio_processor's process_audio method as the callback
        audio_capture.start_stream(audio_processor.process_audio)
        input("Press Enter to stop recording...\n")  # Keep running until user stops
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Stop the stream and clean up
        audio_capture.stop_stream()

