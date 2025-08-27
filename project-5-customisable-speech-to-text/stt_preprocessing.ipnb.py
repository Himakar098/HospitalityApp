import sounddevice as sd # type: ignore
import numpy as np
import queue
import threading
import torchaudio # type: ignore
import noisereduce as nr
import speech_recognition as sr
from scipy.signal import butter, lfilter
import torch
import IPython.display as ipd  # For audio display in Jupyter

# Set up a queue for live audio streaming
audio_queue = queue.Queue()

# Audio stream parameters
sample_rate = 16000
block_size = 1024  # Number of frames per block

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

# STT transcription using speech_recognition
recognizer = sr.Recognizer()

def transcribe_audio(audio_data_path):
    try:
        # Use speech_recognition to recognize audio
        with sr.AudioFile(audio_data_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

# Callback function to handle audio blocks
def audio_callback(indata, frames, time, status):
    """This function is called for each audio block in the stream"""
    if status:
        print(status, flush=True)
    audio_queue.put(indata.copy())  # Add the recorded block to the queue

# Function to stream and process audio for a fixed time (e.g., 10 seconds)
def stream_audio(duration=10):
    with sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback, blocksize=block_size):
        print(f"Recording for {duration} seconds...")
        sd.sleep(duration * 1000)  # Record for the specified duration

        # Get audio data from the queue and process it
        while not audio_queue.empty():
            audio_chunk = audio_queue.get()

            # Preprocess audio chunk
            processed_audio = process_audio(audio_chunk, sample_rate=sample_rate)

            # Ensure audio is saved in a valid format for STT (16-bit PCM WAV)
            processed_audio = torch.tensor(processed_audio, dtype=torch.float32).unsqueeze(0)
            torchaudio.save('temp_audio.wav', processed_audio, sample_rate, format="wav")

            # Perform STT on the processed audio
            transcription = transcribe_audio('temp_audio.wav')
            if transcription:
                print(f"Transcription: {transcription}")

            # Play the processed audio in Jupyter Notebook
            ipd.display(ipd.Audio('temp_audio.wav', rate=sample_rate))

# Run the function to stream and transcribe
stream_audio(duration=10)  # Stream for 10 seconds
