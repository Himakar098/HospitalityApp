import sounddevice as sd
import numpy as np
import queue
import torch
import torchaudio
import noisereduce as nr
from scipy.signal import butter, lfilter
from scipy.io.wavfile import write
from transformers import pipeline
import IPython.display as ipd  # For audio display in Jupyter

# Set up a queue for live audio streaming
audio_queue = queue.Queue()

# Audio stream parameters
sample_rate = 16000
block_size = 1024  # Number of frames per block

# Load the Whisper small model using Hugging Face Transformers
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)

# Preprocessing functions
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def process_audio(audio_chunk, sample_rate=16000, low_freq=300, high_freq=3400):
    # Convert audio to numpy array
    audio_chunk_np = np.array(audio_chunk, dtype=np.float32)
    
    # Noise reduction
    reduced_noise_audio = nr.reduce_noise(y=audio_chunk_np, sr=sample_rate)
    
    # Apply band-pass filtering to isolate speech
    filtered_audio = bandpass_filter(reduced_noise_audio, low_freq, high_freq, sample_rate)
    
    return filtered_audio

# Function to transcribe audio using Whisper small model
def transcribe_audio_whisper(audio_file):
    try:
        result = whisper_asr(audio_file)
        return result['text']
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

# Callback function to handle audio blocks
def audio_callback(indata, frames, time, status):
    """This function is called for each audio block in the stream"""
    if status:
        print(status, flush=True)
    audio_queue.put(indata.copy())  # Add the recorded block to the queue

# Enhanced function to dynamically select the best available input device (skipping Sound Mapper)
def get_default_input_device():
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        # Skip "Sound Mapper" devices and only choose those with input channels
        if device['max_input_channels'] > 0 and "Sound Mapper" not in device['name']:
            print(f"Using input device: {device['name']} (ID: {i})")
            return i
    raise ValueError("No suitable input device found!")

# Function to stream and process audio for a fixed time (e.g., 10 seconds)
def stream_audio(duration=10):
    # Dynamically get the best input device
    device_id = get_default_input_device()
    
    with sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback, blocksize=block_size, device=device_id):
        print(f"Recording for {duration} seconds...")
        sd.sleep(duration * 1000)  # Record for the specified duration

        # Get audio data from the queue and process it
        while not audio_queue.empty():
            audio_chunk = audio_queue.get()

            # Preprocess audio chunk
            processed_audio = process_audio(audio_chunk, sample_rate=sample_rate)

            # Ensure audio is saved in a valid format for STT (16-bit PCM WAV)
            if isinstance(processed_audio, torch.Tensor):  # Check if the data is a tensor
                if processed_audio.shape[0] == 1:  # Check if the first axis can be squeezed
                    processed_audio_np = processed_audio.squeeze(0).numpy()  # Squeeze only if the first axis is size 1
                else:
                    processed_audio_np = processed_audio.numpy()  # Otherwise, use the data as is
            else:
                processed_audio_np = processed_audio  # If it's already a NumPy array, use it as is

            # Save the audio file in 16-bit PCM WAV format using SciPy
            write('temp_audio.wav', sample_rate, processed_audio_np.astype(np.float32))  # Save using SciPy

            # Perform STT using Whisper model
            transcription = transcribe_audio_whisper('temp_audio.wav')
            if transcription:
                print(f"Transcription: {transcription}")

            # Play the processed audio in Jupyter Notebook (if applicable)
            ipd.display(ipd.Audio('temp_audio.wav', rate=sample_rate))

# Run the function to stream and transcribe
stream_audio(duration=10)  # Stream for 10 seconds
