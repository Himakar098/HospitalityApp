# install the required files in the terminal
# pip install pyaudio whisper transformers optimum torchaudio

import pyaudio
import numpy as np
import torch
import openai
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer
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

# Initialize the Parler-TTS model (optional, if TTS is needed)
tokenizer = AutoTokenizer.from_pretrained('Parler-TTS')
tts_model = T5ForConditionalGeneration.from_pretrained('Parler-TTS')
tts_model_quantized = quantize_model_dynamic(tts_model)

# Set up PyAudio for real-time audio streaming
p = pyaudio.PyAudio()

# Audio stream parameters
RATE = 16000  # Sample rate
CHUNK = 1024  # Number of frames per buffer

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
        audio_data = stream.read(CHUNK)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Voice Activity Detection using webrtcvad
        is_speech = vad.is_speech(get_pcm_audio(audio_np, RATE), sample_rate=RATE)

        if is_speech:
            # STT: Convert detected speech to text using Whisper
            inputs = processor(audio_np, return_tensors="pt", sampling_rate=RATE)
            logits = whisper_model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
            print("Transcription:", transcription[0])

            # Optional: Process transcription with GPT-4
            processed_text = generate_text_with_gpt4(transcription[0])
            print("Processed Text:", processed_text)

            # Optional: Convert processed text to speech using TTS
            input_ids = tokenizer(processed_text, return_tensors='pt').input_ids
            speech_output = tts_model_quantized.generate(input_ids)
            
            # Convert the generated speech to audio (assuming `speech_output` is raw audio data)
            # Adjust as needed based on TTS model output format
            audio_np = speech_output.squeeze().numpy()  # Replace with appropriate conversion

            # Play back the generated speech
            play_audio_sd(audio_np, RATE)

except KeyboardInterrupt:
    print("Stopping...")
    stream.stop_stream()
    stream.close()
    p.terminate()

