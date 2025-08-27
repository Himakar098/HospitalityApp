from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import sounddevice as sd
import numpy as np
from scipy.signal import butter, lfilter
import openai
import torch
import langid  # For language identification
import time

# Setup device and model details
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float32

model_id = "distil-whisper/distil-large-v3"

# Load the Distil-Whisper large model and processor
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

# Create a speech recognition pipeline with chunk processing
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=25,
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device,
)

# OpenAI API setup
openai.api_key = 'YOUR_OPENAI_API_KEY'  # Replace with your OpenAI API key

class AudioCapture:
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_stream = None

    def start_stream(self, callback):
        """Starts the audio stream with the given callback for processing."""
        self.audio_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.chunk_size,
            callback=callback
        )
        self.audio_stream.start()
        print("Audio stream started...")

    def stop_stream(self):
        """Stops the audio stream."""
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            print("Audio stream stopped.")

class AudioProcessor:
    def __init__(self, sample_rate=16000, batch_size=3):
        self.sample_rate = sample_rate
        self.transcriptions = []  # To accumulate transcriptions for batch processing
        self.batch_size = batch_size

    def process_audio(self, indata, frames, time, status):
        """Processes the audio input from the stream callback."""
        if status:
            print(f"Stream status: {status}")
        audio_data = np.frombuffer(indata, dtype=np.float32)
        processed_audio = self.noise_reduction(audio_data)

        # Transcribe the processed audio
        transcription = self.transcribe_audio(processed_audio)
        print(f"Whisper Transcription: {transcription}")  # Display the transcription from Whisper

        # Identify the language of the transcription
        language, _ = langid.classify(transcription)
        print(f"Identified Language: {language}")

        # Accumulate transcriptions for batch processing
        self.transcriptions.append(transcription)

        if len(self.transcriptions) >= self.batch_size:
            print("Processing batch transcriptions with GPT-4...")
            # Process transcriptions in a batch to reduce API calls
            full_transcription = " ".join(self.transcriptions)
            response = self.get_gpt_response(full_transcription)
            print(f"GPT-4 Response: {response}")

            # Clear the transcriptions list after processing
            self.transcriptions = []

    def noise_reduction(self, audio_data):
        """Applies a low-pass filter for noise reduction."""
        return self.low_pass_filter(audio_data)

    def low_pass_filter(self, data, cutoff=1000):
        """Basic low-pass filter implementation."""
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(1, normal_cutoff, btype='low', analog=False)
        y = lfilter(b, a, data)
        return y

    def transcribe_audio(self, audio_data):
        """Transcribes the audio data using Whisper pipeline."""
        # Process audio with the pipeline
        transcription = pipe(audio_data)["text"]
        return transcription

    def get_gpt_response(self, transcription):
        """Gets a response from GPT-4 based on the transcription."""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": transcription}
                ]
            )
            return response['choices'][0]['message']['content']
        except openai.error.RateLimitError as e:
            print(f"Rate limit error: {e}")
            time.sleep(60)  # Wait for a minute before retrying
            return "Rate limit reached. Please try again later."

# Main execution
if __name__ == "__main__":
    audio_processor = AudioProcessor(sample_rate=16000, batch_size=3)  # Adjust batch size if needed
    audio_capture = AudioCapture(sample_rate=16000, chunk_size=1024)

    try:
        audio_capture.start_stream(audio_processor.process_audio)
        input("Press Enter to stop recording...\n")  # Keep running until user stops
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        audio_capture.stop_stream()
