import logging
import torch
import torchaudio
import noisereduce as nr
import sounddevice as sd
import numpy as np
from queue import Queue
from threading import Thread
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

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
        self.pause_duration = pause_duration  # Minimum pause duration to split audio (in seconds)
        self.min_chunk_size = min_chunk_size  # Minimum size of audio chunk (samples)
        self.channels = 1
        self.audio_queue = Queue()  # Queue for chunks of audio
        self.is_recording = False
        self.current_chunk = []  # Store the current chunk of audio until a pause is detected

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
        # Convert incoming audio to float32 for consistency
        indata = indata.astype(np.float32)

        # Calculate energy of the audio (signal strength) to detect pauses
        energy = np.linalg.norm(indata)

        # Threshold to detect a pause (30ms of silence)
        pause_threshold = 0.003  # Adjust this based on microphone sensitivity

        if energy > pause_threshold:
            # Add audio to the current chunk if no pause is detected
            self.current_chunk.append(indata)
        else:
            # If a pause is detected and chunk size is sufficient, concatenate and push the chunk to the queue
            if self.current_chunk and len(np.concatenate(self.current_chunk)) >= self.min_chunk_size:
                chunk_data = np.concatenate(self.current_chunk, axis=0)
                self.audio_queue.put(chunk_data)
                self.current_chunk = []  # Reset for the next chunk

    def start_recording(self, device_name=None): 
        """Starts the audio stream and continues recording until stopped."""
        self.is_recording = True
        logging.info(f"Starting audio stream, splitting based on {self.pause_duration * 1000} ms pause detection.")

        # Dynamically select a valid input device (microphone)
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
        self.frame_size = frame_size  # Frame size to split the chunk into smaller pieces

    def process_chunk(self, chunk):
        """Process each chunk of audio."""
        waveform_np = np.array(chunk, dtype=np.float32)  # Ensure float32 to reduce memory usage

        # Ensure mono audio (single channel) for transcription
        if waveform_np.ndim > 1:
            waveform_np = np.mean(waveform_np, axis=1)

        # Apply noise reduction
        reduced_noise_waveform = nr.reduce_noise(y=waveform_np, sr=16000)

        return reduced_noise_waveform

# Step 3: Speech-to-Text Transcription
class SpeechToTextTranscriber:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float32
        model_id = "openai/whisper-small"

        # Load the Whisper model and processor
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, use_safetensors=True
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Create a speech recognition pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=device
        )

    def transcribe_chunk(self, chunk):
        """Transcribe each chunk of processed audio."""
        logging.info(f"Transcribing audio chunk of size {len(chunk)}")
        result = self.pipe(chunk, return_timestamps="word")
        return result['text']

# Main function that handles recording, processing, and transcription in real-time
def main():
    # Initialize components
    recorder = AudioRecorder()
    processor = AudioProcessor()
    transcriber = SpeechToTextTranscriber()

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

                # Transcribe the chunk (in order)
                transcription = transcriber.transcribe_chunk(processed_chunk)
                logging.info(f"Transcription: {transcription}")

                # Mark the queue task as done (important for queue management)
                recorder.audio_queue.task_done()

        # Ensure the last chunk is processed after recording stops
        while not recorder.audio_queue.empty():
            audio_chunk = recorder.audio_queue.get()
            processed_chunk = processor.process_chunk(audio_chunk)
            transcription = transcriber.transcribe_chunk(processed_chunk)
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
