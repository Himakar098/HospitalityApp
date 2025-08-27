import os
import logging
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import torch
from concurrent.futures import ThreadPoolExecutor
from speechbrain.inference.classifiers import EncoderClassifier
from unidecode import unidecode
import torchaudio
import fasttext
from huggingface_hub import hf_hub_download
from queue import Queue
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import deque

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
            self.transcriber.wait_for_completion()

class AudioProcessor:
    def __init__(self, sample_rate=16000, chunk_size=1024, silence_duration=0.1, silence_threshold=0.01):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_duration = silence_duration
        self.silence_threshold = silence_threshold
        self.segment_number = 0
        self.current_audio = np.array([], dtype=np.float32)
        self.silence_count = 0
        self.transcriber = SpeechToTextTranscriber()
        self.audio_queue = Queue()

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

        # Segment audio more aggressively (e.g., every 0.25 seconds)
        if len(self.current_audio) >= self.sample_rate * 0.25:
            self.audio_queue.put(self.current_audio.copy())  # Put a copy of the audio data
            self.segment_number += 1
            self.current_audio = np.array([], dtype=np.float32)  # Clear buffer

        # Trigger transcription from the queue in a separate thread
        if not self.audio_queue.empty():
            self.transcriber.executor.submit(self.transcribe_from_queue)

    def transcribe_from_queue(self):
        """Transcribes audio segments from the queue."""
        while not self.audio_queue.empty():
            audio_data = self.audio_queue.get()
            self.transcriber.transcribe_real_time(audio_data)
 
class LanguageIdentificationNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LanguageIdentificationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
 
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x
 
class EnhancedLanguageIdentifier:
    def __init__(self, fasttext_model, voxlingua_model, mms_model, mms_processor):
        self.fasttext_model = fasttext_model
        self.voxlingua_model = voxlingua_model
        self.mms_model = mms_model
        self.mms_processor = mms_processor
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the neural network
        self.network = LanguageIdentificationNetwork(3, 10, 7).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Define regions and countries
        self.regions = ['Asia', 'Europe', 'Africa', 'North America', 'South America', 'Oceania', 'Middle East']
        self.countries = {
            'Asia': ['China', 'Japan', 'Korea', 'India', 'Thailand', 'Vietnam'],
            'Europe': ['France', 'Germany', 'Italy', 'Spain', 'Russia', 'UK'],
            'Africa': ['Nigeria', 'Kenya', 'South Africa', 'Egypt', 'Morocco'],
            'North America': ['USA', 'Canada', 'Mexico'],
            'South America': ['Brazil', 'Argentina', 'Colombia', 'Peru'],
            'Oceania': ['Australia', 'New Zealand', 'Fiji'],
            'Middle East': ['Saudi Arabia', 'UAE', 'Iran', 'Turkey']
        }
        
        # Create label encoders
        self.region_encoder = LabelEncoder()
        self.region_encoder.fit(self.regions)
        
        # Initialize history
        self.history = deque(maxlen=10)  # Store last 10 predictions
 
    def identify_language(self, audio_data, text):
        # Get predictions from each model
        fasttext_pred = self.fasttext_model.predict(text, k=1)[0][0]
        voxlingua_pred = self.voxlingua_model.classify_batch(audio_data)[3]
        
        # Process audio for MMS model
        inputs = self.mms_processor(audio_data, return_tensors="pt", sampling_rate=16000)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            logits = self.mms_model(**inputs).logits
        mms_pred = self.mms_model.config.id2label[torch.argmax(logits, dim=-1).item()]
        
        # Prepare input for neural network
        input_tensor = torch.tensor([
            [1.0 if fasttext_pred == lang else 0.0 for lang in [fasttext_pred, voxlingua_pred, mms_pred]],
            [1.0 if voxlingua_pred == lang else 0.0 for lang in [fasttext_pred, voxlingua_pred, mms_pred]],
            [1.0 if mms_pred == lang else 0.0 for lang in [fasttext_pred, voxlingua_pred, mms_pred]]
        ], dtype=torch.float32).to(self.device)
        
        # Get prediction from neural network
        with torch.no_grad():
            output = self.network(input_tensor)
        
        predicted_region_idx = torch.argmax(output).item()
        predicted_region = self.regions[predicted_region_idx]
        
        # Update history
        self.history.append(predicted_region)
        
        # Determine most common region in history
        most_common_region = max(set(self.history), key=self.history.count)
        
        # Determine country based on most common region
        possible_countries = self.countries[most_common_region]
        predicted_country = np.random.choice(possible_countries)  # Simplified country selection
        
        return fasttext_pred, predicted_region, predicted_country
 
    def train(self, true_region):
        if len(self.history) > 0:
            true_label = self.region_encoder.transform([true_region])[0]
            predicted_label = self.region_encoder.transform([self.history[-1]])[0]
            
            loss = self.criterion(torch.tensor([[predicted_label]], dtype=torch.float32).to(self.device),
                                  torch.tensor([true_label], dtype=torch.long).to(self.device))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
 
# Update SpeechToTextTranscriber class
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

        self.executor = ThreadPoolExecutor(max_workers=1)
        
        self.enhanced_language_identifier = EnhancedLanguageIdentifier(
            self.fasttext_model, self.audio_language_model, self.mms_lid_model, self.mms_lid_processor
        )
 
    def load_fasttext_model(self):
        """Loads the FastText model for language identification."""
        fasttext_model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        return fasttext.load_model(fasttext_model_path)

    def transcribe_real_time(self, audio_data):
        try:
            # Convert audio_data to the correct format for VoxLingua
            audio_tensor = torch.tensor(audio_data).unsqueeze(0).float()
 
            # Perform language identification
            transcription = self.processor.batch_decode(
                self.model.generate(self.processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)),
                skip_special_tokens=True
            )[0]
 
            detected_language, predicted_region, predicted_country = self.enhanced_language_identifier.identify_language(audio_tensor, transcription)
 
            logging.info(f"Detected Language: {detected_language}")
            logging.info(f"Predicted Region: {predicted_region}")
            logging.info(f"Predicted Country: {predicted_country}")
            logging.info(f"Transcription: {transcription}")
 
            # Here you would typically send the detected language to the Whisper model for transcription
            # For demonstration, we'll just use the existing transcription
 
            # You can train the network here if you have the true region
            # self.enhanced_language_identifier.train(true_region)
 
        except Exception as e:
            logging.error(f"Error in real-time transcription: {e}")
 
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
                # Ensure transcription is in English letters (Romanized form)
                if detected_language_list[0] != "en":
                    transcription = self.romanize_text(transcription, detected_language_list[0])
                    logging.info(f"Transcription in English letters: {transcription}")

            # Detect language from the text using FastText
            detected_text_language = self.identify_language(transcription)
            logging.info(f"Text-based language detection (FastText): {detected_text_language}")

        except Exception as e:
            logging.error(f"Error in real-time transcription: {e}")

    def detect_language_from_audio(self, audio_data):
        """Detects the language of the given audio data directly."""
        try:
            # Convert numpy array to temporary audio file
            temp_audio_file = "temp_audio_for_lang_detection.wav"
            write(temp_audio_file, 16000, np.int16(audio_data / np.max(np.abs(audio_data)) * 32767))

            audio_input, sample_rate = torchaudio.load(temp_audio_file)
            os.remove(temp_audio_file)  # Remove the temporary file

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
        except Exception as e:
            logging.error(f"Error detecting language from audio: {e}")
            return None

    def romanize_text(self, text, language_code):
        """Converts text to English letters (Romanized form) if necessary."""
        if language_code != 'en':
            return unidecode(text)
        else:
            return text  
        
    def identify_language(self, text):
        """Identifies the language of the given text using FastText."""
        predictions = self.fasttext_model.predict(text, k=1)
        return predictions[0][0]

    def wait_for_completion(self):
        """Waits for all transcription threads to complete."""
        self.executor.shutdown(wait=True)


# Main execution
if __name__ == "__main__":
    # Initialize audio capture
    audio_capture = AudioCapture()
    audio_processor = AudioProcessor()

    try:
        # Start recording and processing
        audio_capture.start_stream(audio_processor.process_audio)
        input("Press Enter to stop recording...\n")  # User stops recording manually
    finally:
        # Ensure the stream is stopped and processed properly
        audio_capture.stop_stream()