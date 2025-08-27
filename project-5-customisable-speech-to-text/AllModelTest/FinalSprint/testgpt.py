import os
import logging
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import tempfile
from speechbrain.inference.classifiers import EncoderClassifier
from huggingface_hub import hf_hub_download
import fasttext

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
        self.audio_stream = None
        self.transcriber = SpeechToTextTranscriber()  # Instantiate transcriber
        self.temp_files = []
        self.file_counter = 0

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

    def transcribe_audio(self, audio_data):
        """Transcribes the audio data directly from numpy array."""
        # Convert to mono and normalize to 16-bit PCM
        audio_data_mono = np.mean(audio_data, axis=1) if audio_data.ndim > 1 else audio_data
        audio_data_int16 = np.int16(audio_data_mono / np.max(np.abs(audio_data_mono)) * 32767)
        
        # Generate a unique file name
        self.file_counter += 1
        file_name = f"audio_segment_{self.file_counter}.wav"
        
        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        write(temp_file.name, self.sample_rate, audio_data_int16)
        self.temp_files.append(temp_file.name)
        
        # Start a new thread to handle transcription
        self.transcriber.transcribe(audio_data_int16, file_name)

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
    def __init__(self, sample_rate=16000, chunk_size=1024, silence_duration=1, silence_threshold=0.01):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_duration = silence_duration
        self.silence_threshold = silence_threshold
        self.current_audio = np.array([], dtype=np.float32)
        self.silence_count = 0
        self.MIN_AUDIO_LENGTH = self.sample_rate * 0.5  # 0.5 seconds
        self.MAX_AUDIO_LENGTH = self.sample_rate * 30  # 30 seconds maximum segment length

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

        # If silence persists for the specified duration or max length reached, process the audio segment
        if (self.silence_count >= (self.silence_duration * self.sample_rate / self.chunk_size) or 
            len(self.current_audio) >= self.MAX_AUDIO_LENGTH):
            if len(self.current_audio) > self.MIN_AUDIO_LENGTH:
                audio_capture.transcribe_audio(self.current_audio)
            self.current_audio = np.array([], dtype=np.float32)  # Clear after processing
            self.silence_count = 0

class SpeechToTextTranscriber:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        torch_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32

        model_id = "openai/whisper-large-v3"

        # Load the Whisper model and processor
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
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

        # Load the VoxLingua107 model for audio-based language detection
        self.audio_language_model = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")

        self.fasttext_model = self.load_fasttext_model()
        self.executor = ThreadPoolExecutor(max_workers=2)
        # Define a mapping of VoxLingua107 language codes to Whisper language codes
        self.language_mapping = {
            "en": "english", "zh": "chinese", "de": "german", "es": "spanish", "ru": "russian",
            "ko": "korean", "fr": "french", "ja": "japanese", "pt": "portuguese", "tr": "turkish",
            "pl": "polish", "ca": "catalan", "nl": "dutch", "ar": "arabic", "sv": "swedish",
            "it": "italian", "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
            "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay", "cs": "czech",
            "ro": "romanian", "da": "danish", "hu": "hungarian", "ta": "tamil", "no": "norwegian",
            "th": "thai", "ur": "urdu", "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian",
            "la": "latin", "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak",
            "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali", "sr": "serbian",
            "az": "azerbaijani", "sl": "slovenian", "kn": "kannada", "et": "estonian", "mk": "macedonian",
            "br": "breton", "eu": "basque", "is": "icelandic", "hy": "armenian", "ne": "nepali",
            "mn": "mongolian", "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
            "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala", "km": "khmer",
            "sn": "shona", "yo": "yoruba", "so": "somali", "af": "afrikaans", "oc": "occitan",
            "ka": "georgian", "be": "belarusian", "tg": "tajik", "sd": "sindhi", "gu": "gujarati",
            "am": "amharic", "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese",
            "ht": "haitian creole", "ps": "pashto", "tk": "turkmen", "nn": "nynorsk", "mt": "maltese",
            "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar", "bo": "tibetan", "tl": "tagalog",
            "mg": "malagasy", "as": "assamese", "tt": "tatar", "haw": "hawaiian", "ln": "lingala",
            "ha": "hausa", "ba": "bashkir", "jv": "javanese", "su": "sundanese"
        }

    def _transcribe_audio(self, audio_data, file_name):
        """Internal method to handle transcription and language identification."""
        try:
            logging.info(f"Transcribing audio file: {file_name}")
            generate_kwargs = {
                "return_timestamps": False,
            }

            # Detect language from audio first
            detected_language_code = self.detect_language_from_audio(audio_data)
            logging.info(f"Detected language from audio: {detected_language_code}")

            whisper_language = self.language_mapping.get(detected_language_code.split(':')[0])

            if whisper_language:
                logging.info(f"Mapped to Whisper language: {whisper_language}")
                generate_kwargs["language"] = whisper_language
            else:
                logging.info("Detected language not in mapping. Letting Whisper auto-detect.")

            result = self.pipe(audio_data, generate_kwargs=generate_kwargs)
            text = result['text']

            logging.info(f"Transcription for {file_name}: {text}")
            language = self.identify_language(text)
            logging.info(f"Language detected: {language}")
            return text, whisper_language or detected_language_code

        except Exception as e:
            logging.error(f"Error transcribing audio {file_name}: {e}")
            return f"Error transcribing audio: {e}", "unknown"

    def detect_language_from_audio(self, audio_data):
        """Detects the language of the given audio data directly."""
        try:
            signal = torch.from_numpy(audio_data).float()
            prediction = self.audio_language_model.classify_batch(signal)
            audio_language = prediction[3][0]
            return audio_language
        except Exception as e:
            logging.error(f"Error detecting language from audio: {e}")
            return "unknown"

    def load_fasttext_model(self):
        """Load the FastText language identification model."""
        model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        return fasttext.load_model(model_path)
    def identify_language(self, text):
        """Identifies the language of the given text using FastText."""
        predictions = self.fasttext_model.predict(text, k=1)
        return predictions[0][0]

    def transcribe(self, audio_data, file_name):
        """Transcribes the given audio data and verifies the language in a separate thread."""
        self.executor.submit(self._transcribe_audio, audio_data, file_name)

    def wait_for_completion(self):
        """Waits for all transcription threads to complete."""
        self.executor.shutdown(wait=True)

# Main execution
if __name__ == "__main__":
    sample_rate = 16000
    chunk_size = 1024
    silence_duration = 1
    silence_threshold = 0.01

    audio_processor = AudioProcessor(sample_rate=sample_rate, chunk_size=chunk_size, silence_duration=silence_duration, silence_threshold=silence_threshold)
    audio_capture = AudioCapture(sample_rate=sample_rate, chunk_size=chunk_size, silence_threshold=silence_threshold, silence_duration=silence_duration)

    try:
        audio_capture.start_stream(audio_processor.process_audio)
        input("Press Enter to stop recording...\n")
    finally:
        audio_capture.stop_stream()
        audio_capture.play_or_delete_options()
