from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from concurrent.futures import ThreadPoolExecutor
import fasttext
from huggingface_hub import hf_hub_download
from speechbrain.inference.classifiers import EncoderClassifier
from unidecode import unidecode
import logging


class SpeechToTextTranscriber:
    def __init__(self):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "mps" else torch.float32

        model_id = "openai/whisper-large-v3"

        # Load the Whisper model and processor
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="sdpa"
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Create a speech recognition pipeline without forcing a specific language initially
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
        self.executor = ThreadPoolExecutor(max_workers=1)

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

    def transcribe(self, file_path):
        """Submits the transcription task to the ThreadPoolExecutor."""
        self.executor.submit(self._transcribe, file_path)

    def _transcribe(self, file_path):
        """Performs the transcription of the provided audio file."""
        
        try:
            # Detect language from audio using VoxLingua
            detected_language_list = self.detect_language_from_audio(file_path)


            # Extract the language code from the list (assuming it's the first element)
            if detected_language_list:
                detected_language = detected_language_list[0].split(":")[0]  # Extract code before the colon

                if detected_language in self.language_mapping:
                    whisper_language = self.language_mapping[detected_language]
                    if whisper_language in self.language_mapping.values():
                        logging.info(f"Language detected: {whisper_language}. Forcing Whisper transcription in this language.")
                        result = self.pipe(file_path, generate_kwargs={"return_timestamps": True, "language": whisper_language})
                    else:
                        logging.warning(f"Unsupported language: {whisper_language}. Falling back to Whisper's own detection.")
                        # Let Whisper detect the language automatically
                        result = self.pipe(file_path, generate_kwargs={"return_timestamps": True})
            else:
                logging.warning("VoxLingua could not reliably detect the language. Falling back to Whisper's own detection.")

                    # Let Whisper detect the language automatically
                result = self.pipe(file_path, generate_kwargs={"return_timestamps": True})
            
            # Get transcription text
            transcription = result['text']
            logging.info(f"Transcription for {file_path}: {transcription}")
            logging.info(f"Direct audio language detection: {detected_language_list}")
            language = self.identify_language(transcription)
            logging.info(f"Transcribed text language detection: {language}")
            
            # Ensure transcription is in English letters (Romanized form)
            if detected_language and detected_language != "en":
                transcription = self.romanize_text(transcription, detected_language)
                logging.info(f"Transcription in English letters: {transcription}")

        except Exception as e:
            logging.error(f"Error in transcription: {e}")

    def detect_language_from_audio(self, file_path):
        """Detects the language of the given audio file directly."""
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
        # Placeholder for romanization logic
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
