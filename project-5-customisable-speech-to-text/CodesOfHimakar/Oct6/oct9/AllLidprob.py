import os
import logging
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import torch
from concurrent.futures import ThreadPoolExecutor
from speechbrain.inference.classifiers import EncoderClassifier
from unidecode import unidecode
import torchaudio
import torch.nn.functional as F  # Import softmax for probability calculations
import whisper
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress FutureWarnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")  # Suppress a specific message

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
        torch_dtype = torch.float32 if torch.cuda.is_available() else torch.float32

        # Load Whisper model for transcription
        model_id = "openai/whisper-small"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, 
            use_safetensors=True, attn_implementation="sdpa").to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)

        #load model for Whisper lang detection
        self.wl_model = whisper.load_model("tiny")

        # Load MMS model for language identification (on CPU to prevent OOM issues)
        self.mms_lid_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "facebook/mms-lid-126", torch_dtype=torch_dtype, low_cpu_mem_usage=True, 
            use_safetensors=True).to(self.device) 
        self.mms_lid_processor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-126")

        # Load SpeechBrain VoxLingua for audio language identification
        self.audio_language_model = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")
        
        self.overall_language_weights = {}

        self.executor = ThreadPoolExecutor(max_workers=1)  # Increase workers for concurrent transcriptions

    def transcribe_real_time(self, audio_input):
        """Performs real-time transcription on the provided audio data."""
        try:
            # Detect language from the audio using Whisper Tiny
            detected_language_list_tiny = self.detect_language_from_whisper_tiny(audio_input)
            logging.info(f"Detected Language by tiny: {detected_language_list_tiny}")

            # Detect language from the audio using VoxLingua
            detected_language_list_vox = self.detect_language_from_voxlingua(audio_input)
            logging.info(f"Language detected using VoxLingua: {detected_language_list_vox}")
            # Unfortunately, SpeechBrain's EncoderClassifier for VoxLingua doesn't directly provide probabilities 
            # or a way to get the top k predictions. 
            # It only gives the single most likely language.

            # Detect language from the audio using MMS LID
            detected_language_list_mms = self.detect_language_from_mms(audio_input) 

            # Combine LID results
            final_language, overall_language , final_language_prob = self.combine_lid_results(
                detected_language_list_tiny, detected_language_list_mms, detected_language_list_vox
            )
            logging.info(f"Local Final Language decided: {final_language}")
            logging.info(f"Global Final Language decided: {overall_language}")

            if final_language_prob > 0.75:
                whisper_forced_language = final_language
                logging.info(f"Language decided for whisper: {whisper_forced_language}")
            elif overall_language != 'en':
                whisper_forced_language = overall_language
                logging.info(f"Language decided for whisper: {whisper_forced_language}")
            elif overall_language == 'en' and final_language != 'en':
                whisper_forced_language = final_language
                logging.info(f"Language decided for whisper: {whisper_forced_language}")
            else:
                whisper_forced_language = 'en'
                logging.info(f"Language decided for whisper: {whisper_forced_language}")
            input_features = self.processor(audio_input, sampling_rate=16000, language=whisper_forced_language, return_tensors="pt").input_features.to(self.device)
            predicted_ids = self.model.generate(input_features, output_scores=True, return_dict_in_generate=True)

            # Decode the predicted IDs into text
            transcription = self.processor.batch_decode(predicted_ids.sequences, skip_special_tokens=True)[0]
            logging.info(f"Real-time transcription: {transcription}")

        except RuntimeError as re:
            logging.error(f"CUDA error during transcription: {re}")
        except Exception as e:
            logging.error(f"Error during transcription: {e}")

        if whisper_forced_language != "en":
            transcription = self.romanize_text(transcription)
            logging.info(f"Transcription in English letters: {transcription}")

        logging.info("*********************** End of Transcribing file **********************")

    def combine_lid_results(self, detect_language_from_whisper_tiny, detected_language_list_mms, detected_language_list_vox):
        """Combines language identification results from different models."""

        language_weights = {}

        # --- Whisper Tiny ---
        for language, prob in detect_language_from_whisper_tiny:
            language = language.lower()  # Ensure lowercase
            language_weights[language] = prob  

        # --- VoxLingua ---
        voxlingua_lang, vox_prob = detected_language_list_vox[0]  # Unpack the tuple
        if voxlingua_lang in language_weights:
            language_weights[voxlingua_lang] = max(language_weights[voxlingua_lang], vox_prob)   # Update higher weighted probability if same as Whisper
        else:
            language_weights[voxlingua_lang] = vox_prob  # Use VoxLingua probability

        # --- MMS ---
        for lang, prob in detected_language_list_mms:
            if lang in language_weights:
                language_weights[lang] = max(language_weights[lang], prob)  # Compare and take the higher probability
            else:
                language_weights[lang] = prob  # Add new language with its probability

        # --- Determine final language ---
        final_language = max(language_weights, key=language_weights.get)  # Get key with max value
        final_language_ = language_weights[final_language]  # Round to 4 decimal places
        logging.info(f"Final language weights determined: {language_weights}")

        # --- Update overall language weights ---
        for lang, weight in language_weights.items(): 
            if lang in self.overall_language_weights:
                self.overall_language_weights[lang] += weight  # Add weight to existing language
            else:
                self.overall_language_weights[lang] = weight  # Add new language with its weight
        logging.info(f"Overall language weights: {self.overall_language_weights}")

        # --- Determine overall language ---
        overall_language = max(self.overall_language_weights, key=self.overall_language_weights.get) # Get the key

        return final_language, overall_language, final_language_
    
    def detect_language_from_whisper_tiny(self, audio_input, top_k = 5):
        """Detects language from the provided audio input using VoxLingua."""
        try:
            # Use Whisper Tiny to detect the language and get the top 5 probabilities
            audio = whisper.pad_or_trim(audio_input)  # Prepare the audio for Whisper Tiny
            mel = whisper.log_mel_spectrogram(audio).to(self.wl_model.device)  # Convert to Mel spectrogram

            # Get language probabilities from the Whisper Tiny model
            _, probs = self.wl_model.detect_language(mel)

            # Sort the languages by probability and get the top 5
            sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)[:top_k]
            logging.info("Top 5 detected languages by Whisper Tiny:")
            return sorted_probs
        except Exception as e:
            logging.error(f"Error detecting language with Tiny: {e}")
            return None

    def detect_language_from_voxlingua(self, audio_input):
        """Detects language from the provided audio input using VoxLingua."""
        try:
            # VoxLingua expects a waveform tensor
            audio_tensor = torch.tensor(audio_input).unsqueeze(0)  # Ensure batch dimension
            language_prediction = self.audio_language_model.classify_batch(audio_tensor)
            vox_predicted_language = language_prediction[3][0][:2].lower()
            logits = language_prediction[0]  # Logits for each language
            vox_prob = round(logits.exp().max().item(), 4)
            #logging.info(f"VoxLingua detected language: {predicted_language}")
            vox_language = []
            vox_language.append((vox_predicted_language, vox_prob))
            return vox_language
        except Exception as e:
            logging.error(f"Error detecting language with VoxLingua: {e}")
            return None

    def detect_language_from_mms(self, audio_input, top_k = 5):
        """Detects language from the provided audio input using MMS LID."""
        try:
            # Map for MMS 3-letter tokens to Whisper 2-letter tokens
            mms_to_whisper = {
                "ara": "ar", "cmn": "zh", "eng": "en", "spa": "es", "fra": "fr",
                "mlg": "mg", "swe": "sv", "por": "pt", "vie": "vi", "ful": "Fula",
                "sun": "su", "asm": "as", "ben": "bn", "zlm": "ms", "kor": "ko",
                "ind": "id", "hin": "hi", "tuk": "tk", "urd": "ur", "aze": "az",
                "slv": "sl", "mon": "mn", "hau": "ha", "tel": "te", "swh": "sw",
                "bod": "bo", "rus": "ru", "tur": "tr", "heb": "he", "mar": "mr",
                "som": "so", "tgl": "tl", "tat": "tt", "tha": "th", "cat": "ca",
                "ron": "ro", "mal": "ml", "bel": "be", "pol": "pl", "yor": "yo",
                "nld": "nl", "bul": "bg", "hat": "ht", "afr": "af", "isl": "is",
                "amh": "am", "tam": "ta", "hun": "hu", "hrv": "hr", "lit": "lt",
                "cym": "cy", "fas": "fa", "mkd": "mk", "ell": "el", "bos": "bs",
                "deu": "de", "sqi": "sq", "jav": "jw", "nob": "norwegian bokmål", "uzb": "uz",
                "snd": "sd", "lat": "la", "nya": "chichewa", "grn": "guarani", "mya": "my",
                "orm": "oromo", "lin": "ln", "hye": "hy", "yue": "yue", "pan": "pa",
                "jpn": "ja", "kaz": "kk", "npi": "ne", "kat": "ka", "guj": "gu",
                "kan": "kn", "tgk": "tg", "ukr": "uk", "ces": "cs", "lav": "lv",
                "bak": "ba", "khm": "km", "fao": "fo", "glg": "gl", "ltz": "lb",
                "lao": "lo", "mlt": "mt", "sin": "si", "sna": "sn", "ita": "it",
                "srp": "sr", "mri": "mi", "nno": "nn", "pus": "ps", "eus": "eu",
                "ory": "oriya", "lug": "luganda", "bre": "br", "luo": "luo", "slk": "sk",
                "fin": "fi", "dan": "da", "yid": "yi", "est": "et", "ceb": "cebuano",
                "war": "waray", "san": "sa", "kir": "kyrgyz", "oci": "oc", "wol": "wolof",
                "haw": "haw", "kam": "kamba", "umb": "umbundu", "xho": "xhosa", "epo": "esperanto",
                "zul": "zulu", "ibo": "igbo", "abk": "Abkhazian", "ckb": "kurdish", "nso": "northern sotho",
                "gle": "irish", "kea": "kabuverdianu", "ast": "asturian", "sco": "scots", "glv": "manx",
                "ina": "interlingua"
            }

            audio_features = self.mms_lid_processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values.to(self.device)
            with torch.no_grad():
                logits = self.mms_lid_model(audio_features).logits
                logits = logits - torch.max(logits, dim=-1, keepdim=True).values
                sorted_logits = torch.sort(logits, dim=-1, descending=True)
                top_k_logits = sorted_logits[0][0][:top_k]
                top_k_indices = sorted_logits[1][0][:top_k]
                probabilities = F.softmax(logits, dim=-1)
                top_k_probabilities = probabilities[0][top_k_indices]

                logging.info("Top 5 probable languages (MMS):")
                top_languages = []
                for i in range(top_k):
                    lang = self.mms_lid_model.config.id2label[top_k_indices[i].item()]
                    prob = top_k_probabilities[i].item()
                    # Map MMS language code to Whisper code
                    mapped_lang = mms_to_whisper.get(lang[:3].lower())
                    if mapped_lang:
                        lang = mapped_lang  # Update the lang code
                    logging.info(f"Language: {lang}, Probability: {prob:.4f}")
                    top_languages.append((lang, prob))

                return top_languages
        except Exception as e:
            logging.error(f"Error detecting language from audio: {e}")
            return None
        
    def romanize_text(self, text):
        """Romanizes the text if the detected language is non-Latin."""
        try:
            romanized_text = unidecode(text)
            #logging.info(f"Romanized text: {romanized_text}")
            return romanized_text
        except Exception as e:
            logging.error(f"Error romanizing text: {e}")
            return text
   
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