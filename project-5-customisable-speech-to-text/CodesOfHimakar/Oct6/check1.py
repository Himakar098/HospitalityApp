import os
import logging
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Wav2Vec2ForSequenceClassification,
    AutoFeatureExtractor,
    pipeline,
)
import torch
from concurrent.futures import ThreadPoolExecutor
from speechbrain.inference.classifiers import EncoderClassifier
from unidecode import unidecode
import torchaudio
import torch.nn.functional as F
import whisper
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("transcription.log"),
        logging.StreamHandler()
    ]
)

# Constants
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1
MIN_AUDIO_LENGTH = SAMPLE_RATE * 0.5  # Minimum audio length for transcription (0.5 seconds)

class AudioCapture:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.chunk_size = CHUNK_SIZE
        self.silence_threshold = SILENCE_THRESHOLD
        self.silence_duration = SILENCE_DURATION
        self.temp_files = []
        self.audio_stream = None
        self.transcriber = SpeechToTextTranscriber()  
        self.audio_processor = AudioProcessor(self.transcriber, self)

    def start_stream(self, callback):
        """Starts the audio stream with the given callback for processing."""
        try:
            self.audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.chunk_size,
                callback=callback
            )
            self.audio_stream.start()
            logging.info("Recording started... Speak into your microphone.")
        except Exception as e:
            logging.error(f"Error starting audio stream: {e}")

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
            try:
                os.system(f"ffplay -nodisp -autoexit {file}")
            except Exception as e:
                logging.error(f"Error playing audio file: {e}")

    def delete_files(self):
        """Deletes all saved audio files."""
        for file in self.temp_files:
            try:
                os.remove(file)
                logging.info(f"Deleted {file}.")
            except Exception as e:
                logging.error(f"Error deleting audio file: {e}")
        self.temp_files.clear()


class AudioProcessor:
    def __init__(self, transcriber, audio_capture):
        self.sample_rate = audio_capture.sample_rate
        self.chunk_size = audio_capture.chunk_size
        self.silence_duration = audio_capture.silence_duration
        self.silence_threshold = audio_capture.silence_threshold
        self.segment_number = 0
        self.current_audio = np.array([], dtype=np.float32)
        self.silence_count = 0
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
            if len(self.current_audio) > MIN_AUDIO_LENGTH:
                file_name = self.audio_capture.save_and_transcribe_audio(self.current_audio, self.segment_number)
                if file_name:
                    self.transcriber.transcribe(file_name)
                self.segment_number += 1
                self.current_audio = np.array([], dtype=np.float32)
            self.silence_count = 0


class SpeechToTextTranscriber:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32  # Use float16 for CUDA if available

        # Load Whisper model for transcription
        model_id = "openai/whisper-large-v3-turbo"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, 
            use_safetensors=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device,
        )

        # Load model for Whisper language detection
        self.wl_model = whisper.load_model("tiny")

        # Load MMS model for language identification
        self.mms_lid_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "facebook/mms-lid-126", torch_dtype=torch_dtype, low_cpu_mem_usage=True, 
            use_safetensors=True
        ).to(self.device) 
        self.mms_lid_processor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-126")

        # Load SpeechBrain VoxLingua for audio language identification
        self.audio_language_model = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")
        
        self.overall_language_weights = {}
        self.executor = ThreadPoolExecutor(max_workers=1)  # Use a single worker for sequential processing

    def transcribe_real_time(self, audio_input):
        """Performs real-time transcription on the provided audio data."""
        try:
            # Prepare audio data for Whisper
            input_features = self.processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)

            # Generate token ids
            predicted_ids = self.model.generate(input_features, output_scores=True, return_dict_in_generate=True)

            # Extract the predicted language token
            predicted_language_token = predicted_ids.sequences[0, 1].item() 
            detected_language_whisper = self.processor.tokenizer.convert_ids_to_tokens([predicted_language_token])[0]
            detected_language_whisper = detected_language_whisper.replace("<|", "").replace("|>", "")  
            logging.info(f"Detected language token by Whisper: {detected_language_whisper}")

            # Detect language using Whisper tiny
            result = self.wl_model.transcribe(audio_input)
            detected_language = result['language']
            logging.info(f"Detected Language by tiny: {detected_language}")

            # Detect language from the audio using VoxLingua
            detected_language_list_vox = self.detect_language_from_voxlingua(audio_input)
            logging.info(f"Language detected using VoxLingua: {detected_language_list_vox}")

            # Detect language from the audio using MMS LID
            detected_language_list_mms = self.detect_language_from_mms(audio_input) 

            # Combine LID results
            final_language, overall_language = self.combine_lid_results(
                detected_language_whisper, detected_language_list_mms, detected_language_list_vox
            )
            logging.info(f"Local Language Predictions: Whisper: {detected_language_whisper}, MMS: {detected_language_list_mms}, VoxLingua: {detected_language_list_vox}")
            logging.info(f"Overall detected language: {final_language}")

            # Transcribe the audio in the detected language
            self.transcribe_audio_in_language(audio_input, overall_language)

        except Exception as e:
            logging.error(f"Error during real-time transcription: {e}")

    def detect_language_from_voxlingua(self, audio_input):
        """Detects language using the VoxLingua model."""
        try:
            signal, sample_rate = torchaudio.load(audio_input)
            detected_language_list = self.audio_language_model.classify(signal, sample_rate).argmax(dim=1).tolist()
            return detected_language_list
        except Exception as e:
            logging.error(f"Error detecting language with VoxLingua: {e}")
            return []

    def detect_language_from_mms(self, audio_input):
        """Detects language using the MMS LID model."""
        try:
            audio_input = torchaudio.load(audio_input)[0]
            input_features = self.mms_lid_processor(audio_input, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
            logits = self.mms_lid_model(input_features).logits
            predicted_class_ids = torch.argmax(logits, dim=1)
            detected_language_list = predicted_class_ids.tolist()
            return detected_language_list
        except Exception as e:
            logging.error(f"Error detecting language with MMS LID: {e}")
            return []

    def combine_lid_results(self, detected_language_whisper, detected_language_list_mms, detected_language_list_vox):
        """Combines language detection results from different models."""
        detected_languages = {
            "whisper": detected_language_whisper,
            "mms": detected_language_list_mms,
            "vox": detected_language_list_vox,
        }
        # Here you can add a more complex logic to combine results and determine the final detected language.
        final_language = detected_language_whisper  # Fallback to Whisper's detection
        overall_language = "English"  # Default to English for example; replace with actual logic

        return final_language, overall_language

    def transcribe_audio_in_language(self, audio_input, language):
        """Transcribes audio in the specified language."""
        logging.info(f"Transcribing audio in detected language: {language}")
        
        try:
            # Use the processor to prepare the audio input
            audio_features = self.processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
            result = self.pipe(audio_features, language=language)
            
            transcription_text = result['text']
            logging.info(f"Transcription: {transcription_text}")
        except Exception as e:
            logging.error(f"Error during transcription in {language}: {e}")

    def wait_for_completion(self):
        """Waits for all transcription tasks to complete."""
        self.executor.shutdown(wait=True)

def main():
    audio_capture = AudioCapture()

    try:
        audio_capture.start_stream(audio_capture.audio_processor.process_audio)
        input("Press Enter to stop recording...")
    finally:
        audio_capture.stop_stream()

if __name__ == "__main__":
    main()
