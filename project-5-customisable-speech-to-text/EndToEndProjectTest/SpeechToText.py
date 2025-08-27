import pyaudio
import numpy as np
import torch
import openai
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import webrtcvad
import sounddevice as sd

# Set up OpenAI API key
openai.api_key = 'your_openai_api_key_here'

# Initialize the Whisper model for STT
model_name = "openai/whisper-large"
whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

# Initialize the VAD model (using webrtcvad)
vad = webrtcvad.Vad(1)  # Mode 1 is a good trade-off between sensitivity and false positives

# Set up PyAudio for real-time audio streaming
p = pyaudio.PyAudio()

# Audio stream parameters
RATE = 16000  # Sample rate
CHUNK = 256  # Number of frames per buffer (16 ms frames)

stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening...")

# Function to play audio using sounddevice
def play_audio_sd(audio_np, rate):
    sd.play(audio_np, samplerate=rate)
    sd.wait()  # Wait until the audio is finished playing

# Function to convert raw audio to a format usable by VAD
def get_pcm_audio(audio_np, rate):
    return (audio_np * 32768).astype(np.int16).tobytes()

# Function to call OpenAI's GPT-4 API
def generate_text_with_gpt4(prompt):
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Real-time processing loop
try:
    while True:
        try:
            audio_data = stream.read(CHUNK)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Voice Activity Detection using webrtcvad
            pcm_audio = get_pcm_audio(audio_np, RATE)
            is_speech = vad.is_speech(pcm_audio, sample_rate=RATE)

            if is_speech:
                # STT: Convert detected speech to text using Whisper
                inputs = processor(audio_np, return_tensors="pt", sampling_rate=RATE)
                logits = whisper_model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)
                print("Transcription:", transcription[0])

                # Process transcription with GPT-4
                processed_text = generate_text_with_gpt4(transcription[0])
                print("Processed Text:", processed_text)

        except webrtcvad.Error as e:
            print(f"VAD Error: {e}")

except KeyboardInterrupt:
    print("Stopping...")
    stream.stop_stream()
    stream.close()
    p.terminate()
