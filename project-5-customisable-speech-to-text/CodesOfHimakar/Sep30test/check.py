import os
import logging
import sounddevice as sd
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from concurrent.futures import ThreadPoolExecutor
import fasttext
from huggingface_hub import hf_hub_download
from speechbrain.inference.classifiers import EncoderClassifier
from unidecode import unidecode

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
        self.audio_stream = None
        self.transcriber = SpeechToTextTranscriber()  # Instantiate transcriber
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
        """Stops the audio stream."""
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            logging.info("Recording stopped.")
            logging.info("Waiting for all transcriptions to complete...")
            self.transcriber.wait_for_completion()
            logging.info("All transcriptions are complete.")

class AudioProcessor:
    def __init__(self, sample_rate=16000, chunk_size=1024, silence_duration=0.5, silence_threshold=0.01):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_duration = silence_duration
        self.silence_threshold = silence_threshold
        self.current_audio = np.array([], dtype=np.float32)
        self.silence_count = 0

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

        # If silence persists for the specified duration, transcribe the collected audio
        if self.silence_count >= (self.silence_duration * self.sample_rate / self.chunk_size):
            if len(self.current_audio) > 0:
                # Transcribe the current audio data and clear it
                audio_capture.transcriber.transcribe_audio_segment(self.current_audio)
                self.current_audio = np.array([], dtype=np.float32)  # Clear after transcription
            self.silence_count = 0

class SpeechToTextTranscriber:
    def __init__(self):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "mps" else torch.float32

        model_id = "openai/whisper-large-v3"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="sdpa"
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device
        )

        self.audio_language_model = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")
        self.fasttext_model = self.load_fasttext_model()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.transcribed_words = []  # Collect transcribed words
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
            "ht": "haitian creole", "ps": "pashto", "tk": "turkmen", "nn": "nynorsk",
            "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar", "bo": "tibetan",
            "tl": "tagalog", "mg": "malagasy", "as": "assamese", "tt": "tatar", "haw": "hawaiian",
            "ln": "lingala", "ha": "hausa", "ba": "bashkir", "jw": "javanese", "su": "sundanese"
        }

    def load_fasttext_model(self):
        """Downloads and loads the FastText model for text-based language identification."""
        fasttext_model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        return fasttext.load_model(fasttext_model_path)

    def transcribe_audio_segment(self, audio_data):
        """Transcribes the current audio data in a separate thread."""
        self.executor.submit(self._transcribe_audio_segment, audio_data)

    def _transcribe_audio_segment(self, audio_data):
        """Processes and transcribes a segment of audio."""
        try:
            # Convert audio data to 16-bit PCM
            audio_data_int16 = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)

            # Detect language from the audio
            detected_language_list = self.detect_language_from_audio(audio_data_int16)

            # Use detected language to transcribe
            if detected_language_list:
                detected_language = detected_language_list[0].split(":")[0]
                whisper_language = audio_capture.transcriber.language_mapping.get(detected_language, "auto")
                result = self.pipe(audio_data_int16, generate_kwargs={"return_timestamps": True, "language": whisper_language})
            else:
                result = self.pipe(audio_data_int16, generate_kwargs={"return_timestamps": True})

            # Collect the transcribed words
            transcription = result['text']
            transcribed_words = transcription.split()

            # Display words in sentences of 5 words each
            self.transcribed_words.extend(transcribed_words)
            while len(self.transcribed_words) >= 3:
                sentence = " ".join(self.transcribed_words[:3])
                logging.info(f"Transcribed sentence: {sentence}")
                print(sentence)
                self.transcribed_words = self.transcribed_words[3:]

        except Exception as e:
            logging.error(f"Error in transcription: {e}")

    def detect_language_from_audio(self, audio_data):
        """Detects the language of the given audio data."""
        try:
            signal = self.audio_language_model.load_audio(audio_data)
            prediction = self.audio_language_model.classify_batch(signal)
            return prediction[3]  # ISO code of the detected language
        except Exception as e:
            logging.error(f"Error detecting language from audio: {e}")
            return None

    def wait_for_completion(self):
        """Waits for all transcriptions to finish."""
        self.executor.shutdown(wait=True)

# Main execution
if __name__ == "__main__":

    silence_duration = 0.5

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
