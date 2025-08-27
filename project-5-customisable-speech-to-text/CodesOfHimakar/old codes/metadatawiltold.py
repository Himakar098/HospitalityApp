import os
import logging
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
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
                self.audio_capture.save_and_transcribe_audio(self.current_audio, self.segment_number)
                self.segment_number += 1
                self.current_audio = np.array([], dtype=np.float32)  # Clear after saving
            self.silence_count = 0

class SpeechToTextTranscriber:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Load Whisper model for transcription
        model_id = "openai/whisper-small"
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, 
            use_safetensors=True, attn_implementation="sdpa").to(self.device)
        self.processor = WhisperProcessor.from_pretrained(model_id)

        # Load MMS model for language identification (on CPU to prevent OOM issues)
        self.mms_lid_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/mms-lid-126").to(self.device)
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
        """Performs real-time transcription and detects language on the provided audio data."""
        try:
            # Load the audio data from the file
            audio_data, samplerate = torchaudio.load(audio_data)
            audio_data = audio_data.squeeze().numpy()

            # Detect language from the audio using Whisper
            input_features = self.processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)

            # Generate token ids with language detection
            predicted_ids = self.model.generate(input_features)

            # Decode the predicted IDs into text
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            # Extract the language code detected by Whisper
            whisper_detected_language_tokens = self.processor.tokenizer.convert_ids_to_tokens(predicted_ids[0])
            if len(whisper_detected_language_tokens) > 0:
                whisper_detected_language = whisper_detected_language_tokens[0].split('_')[0]
            else:
                whisper_detected_language = "unknown"  # Handle cases where language is not detected

            logging.info(f"Whisper transcription: {transcription}")
            logging.info(f"Whisper detected language: {whisper_detected_language}")

            # Detect language from audio using MMS LID and VoxLingua
            detected_language_list = self.detect_language_from_audio(audio_data, samplerate)
            if detected_language_list:
                logging.info(f"MMS LID detected language: {detected_language_list[0][0]}, probability: {detected_language_list[0][1]:.4f}")
                logging.info(f"VoxLingua detected language: {detected_language_list[1][0]}, probability: {detected_language_list[1][1]:.4f}")

            # If Whisper detected language is not English, romanize the transcription
            if whisper_detected_language != "en":
                transcription = self.romanize_text(transcription, whisper_detected_language)
                logging.info(f"Transcription (Romanized): {transcription}")

            # Detect language from text using FastText
            detected_text_language, confidence = self.identify_language(transcription)
            logging.info(f"Text-based language detection (FastText): {detected_text_language} with probability {confidence:.4f}")

        except Exception as e:
            logging.error(f"Error in real-time transcription: {e}")
    
    def detect_language_from_audio(self, audio_data, samplerate):
        """Detects the spoken language from the audio using MMS LID and SpeechBrain models."""
        detected_languages = []

        # SpeechBrain language identification (VoxLingua)
        language_id, score = self.audio_language_model.classify_batch(torch.tensor(audio_data).unsqueeze(0))
        logging.info(f"SpeechBrain language identification result: {language_id}, probability: {score:.4f}")
        detected_languages.append((language_id[3], score.item()))  # Append ISO code and probability for VoxLingua

        # MMS LID language identification
        input_values = self.mms_lid_processor(torch.from_numpy(audio_data.squeeze()), return_tensors="pt", sampling_rate=samplerate).input_values.to(self.device)
        logits = self.mms_lid_model(input_values).logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        language_code = self.mms_lid_model.config.id2label[predicted_id]
        probability = F.softmax(logits, dim=-1)[0][predicted_id].item()  # Softmax to get probability
        logging.info(f"MMS language identification result: {language_code}, probability: {probability:.4f}")
        detected_languages.append((language_code, probability))  # Append language and probability for MMS LID

        return detected_languages


    def romanize_text(self, transcription, detected_language):
        """Romanizes the text if the detected language is non-Latin script."""
        romanized_transcription = transcription
        if detected_language != "en":
            romanized_transcription = unidecode(transcription)
            logging.info(f"Text romanized from {detected_language} script: {romanized_transcription}")
        return romanized_transcription

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

