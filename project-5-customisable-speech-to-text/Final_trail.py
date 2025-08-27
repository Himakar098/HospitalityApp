import logging
import torch
import torchaudio
import noisereduce as nr
import sounddevice as sd
import numpy as np
from queue import Queue
from threading import Thread
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from speechbrain.inference.classifiers import EncoderClassifier
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import whisper
import torch.nn.functional as F

# Configure logging to handle Unicode characters
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("transcription.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Step 1: Audio Recorder and Queue Manager
class AudioRecorder:
    def __init__(self, sample_rate=16000, pause_duration=0.03, min_chunk_size=16000):
        self.sample_rate = sample_rate
        self.pause_duration = pause_duration
        self.min_chunk_size = min_chunk_size
        self.channels = 1
        self.audio_queue = Queue()
        self.is_recording = False
        self.current_chunk = []

    def list_devices(self):
        """Lists all available audio devices."""
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']}")

    def get_default_input_device(self):
        """Automatically selects the first valid microphone device."""
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"Using input device: {device['name']} (ID: {i})")
                return i
        raise ValueError("No suitable input device found!")

    def audio_callback(self, indata, frames, time, status):
        """This function is called every time a chunk of audio is available."""
        indata = indata.astype(np.float32)

        # Calculate energy of the audio (signal strength) to detect pauses
        energy = np.linalg.norm(indata)
        pause_threshold = 0.003  # Adjust this based on microphone sensitivity

        if energy > pause_threshold:
            self.current_chunk.append(indata)
        else:
            if self.current_chunk and len(np.concatenate(self.current_chunk)) >= self.min_chunk_size:
                chunk_data = np.concatenate(self.current_chunk, axis=0)
                self.audio_queue.put(chunk_data)
                self.current_chunk = []

    def start_recording(self, device_name=None):
        """Starts the audio stream and continues recording until stopped."""
        self.is_recording = True
        logging.info(f"Starting audio stream, splitting based on {self.pause_duration * 1000} ms pause detection.")

        if device_name is None:
            device_id = self.get_default_input_device()
        else:
            device_id = None
            for i, device in enumerate(sd.query_devices()):
                if device_name in device['name']:
                    device_id = i
                    break
            if device_id is None:
                raise ValueError(f"Device '{device_name}' not found")

        with sd.InputStream(callback=self.audio_callback, channels=self.channels, samplerate=self.sample_rate, device=device_id):
            input("Press Enter to stop streaming...\n")
            self.is_recording = False
            logging.info("Stopped recording.")

# Step 2: Audio Processing (processing each chunk for noise reduction and pause detection)
class AudioProcessor:
    def __init__(self, frame_size=1024):
        self.frame_size = frame_size

    def process_chunk(self, chunk):
        """Process each chunk of audio."""
        waveform_np = np.array(chunk, dtype=np.float32)

        # Ensure mono audio (single channel) for transcription
        if waveform_np.ndim > 1:
            waveform_np = np.mean(waveform_np, axis=1)

        # Apply noise reduction
        reduced_noise_waveform = nr.reduce_noise(y=waveform_np, sr=16000)

        return reduced_noise_waveform

# Step 3: Language Identification (LID)
class LanguageIdentifier:
    def __init__(self):
        # Initialize the VoxLingua model
        self.audio_language_model = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")

        # Initialize the MMS LID model and feature extractor
        self.mms_lid_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "facebook/mms-lid-126", low_cpu_mem_usage=True, torch_dtype=torch.float32
        ).to("cpu")
        self.mms_lid_processor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-126")

        # Mapping MMS codes to Whisper codes
        self.mms_to_whisper_mapping = {
            "ara": "ar", "cmn": "zh", "eng": "en", "spa": "es", "fra": "fr",
            "hin": "hi", "rus": "ru", "kor": "ko", "por": "pt", "tur": "tr",
            # Add more language mappings as necessary
        }

    def detect_language(self, audio_input):
        """Detects language using both VoxLingua and MMS models."""
        vox_language, vox_prob = self.detect_language_from_voxlingua(audio_input)
        mms_language, mms_prob = self.detect_language_from_mms(audio_input)

        combined_language = self.combine_lid_results(vox_language, vox_prob, mms_language, mms_prob)
        return combined_language

    def detect_language_from_voxlingua(self, audio_input):
        """Detects language using VoxLingua."""
        audio_tensor = torch.tensor(audio_input).unsqueeze(0)
        language_prediction = self.audio_language_model.classify_batch(audio_tensor)
        vox_language = language_prediction[3][0][:2].lower()
        vox_prob = round(language_prediction[0].exp().max().item(), 4)
        return vox_language, vox_prob

    def detect_language_from_mms(self, audio_input):
        """Detects language using MMS LID."""
        audio_features = self.mms_lid_processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values.to("cpu")
        logits = self.mms_lid_model(audio_features).logits
        probabilities = F.softmax(logits, dim=-1)
        mms_language = self.mms_lid_model.config.id2label[torch.argmax(probabilities).item()][:3].lower()
        mms_prob = round(torch.max(probabilities).item(), 4)

        mms_language_whisper = self.mms_to_whisper_mapping.get(mms_language, "en")
        return mms_language_whisper, mms_prob

    def combine_lid_results(self, vox_language, vox_prob, mms_language, mms_prob):
        """Combines the results from VoxLingua and MMS models."""
        language_weights = {vox_language: vox_prob, mms_language: mms_prob}
        final_language = max(language_weights, key=language_weights.get)
        logging.info(f"Final detected language: {final_language} (Confidence: {language_weights[final_language]})")
        return final_language

# Step 4: Speech-to-Text Transcription
class SpeechToTextTranscriber:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "openai/whisper-small"

        # Load the Whisper model and processor
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Create a speech recognition pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=device
        )

    def transcribe_chunk(self, chunk, detected_language):
        """Transcribe each chunk of processed audio."""
        logging.info(f"Transcribing audio chunk in language: {detected_language}")

        # Force Whisper to use the detected language
        forced_language = {"language": detected_language}

        # Perform transcription without 'language' as a pipeline argument
        result = self.pipe(chunk, return_timestamps="word")
        
        return result['text']


# Main function that handles recording, processing, LID, and transcription in real-time
def main():
    # Initialize components
    recorder = AudioRecorder()
    processor = AudioProcessor()
    transcriber = SpeechToTextTranscriber()
    lid = LanguageIdentifier()

    # Start recording in a separate thread
    recording_thread = Thread(target=recorder.start_recording)
    recording_thread.start()

    # Function to process the queue in the order chunks were added
    def process_queue():
        while recorder.is_recording or not recorder.audio_queue.empty():
            if not recorder.audio_queue.empty():
                # Get the next chunk from the queue
                audio_chunk = recorder.audio_queue.get()

                # Process the chunk (remove noise)
                processed_chunk = processor.process_chunk(audio_chunk)

                # Detect language using LID models
                detected_language = lid.detect_language(processed_chunk)

                                # Transcribe the chunk (in the detected language)
                transcription = transcriber.transcribe_chunk(processed_chunk, detected_language)
                logging.info(f"Transcription: {transcription}")

                # Mark the queue task as done
                recorder.audio_queue.task_done()

        # Ensure the last chunk is processed after recording stops
        while not recorder.audio_queue.empty():
            audio_chunk = recorder.audio_queue.get()
            processed_chunk = processor.process_chunk(audio_chunk)
            detected_language = lid.detect_language(processed_chunk)
            transcription = transcriber.transcribe_chunk(processed_chunk, detected_language)
            logging.info(f"Final Transcription: {transcription}")
            recorder.audio_queue.task_done()

    # Start the queue processing in the same thread to ensure order
    queue_processing_thread = Thread(target=process_queue)
    queue_processing_thread.start()

    # Wait for the recording thread to finish
    recording_thread.join()

    # Wait for the queue processing to finish
    queue_processing_thread.join()

if __name__ == "__main__":
    main()


