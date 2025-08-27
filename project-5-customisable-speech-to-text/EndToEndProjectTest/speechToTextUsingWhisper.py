import sounddevice as sd
import numpy as np
from scipy.signal import butter, lfilter
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch

# Load Whisper model and processor
Whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
Whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

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
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def process_audio(self, indata, frames, time, status):
        """Processes the audio input from the stream callback."""
        if status:
            print(f"Stream status: {status}")
        audio_data = np.frombuffer(indata, dtype=np.float32)
        processed_audio = self.noise_reduction(audio_data)

        # Transcribe the processed audio
        transcription = self.transcribe_audio(processed_audio)
        print(f"Transcription: {transcription}")

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
        """Transcribes the audio data using Whisper."""
        # Preprocess audio for Whisper model
        inputs = Whisper_processor(audio_data, sampling_rate=self.sample_rate, return_tensors="pt")
        audio_tensor = inputs.input_features  # 'input_features' instead of 'input_values'

        # Ensure the model is in evaluation mode
        Whisper_model.eval()

        # Generate transcription
        with torch.no_grad():
            output = Whisper_model.generate(audio_tensor)

        # Decode transcription
        transcription = Whisper_processor.decode(output[0], skip_special_tokens=True)
        return transcription

# Main execution
if __name__ == "__main__":
    audio_processor = AudioProcessor(sample_rate=16000)
    audio_capture = AudioCapture(sample_rate=16000, chunk_size=1024)

    try:
        audio_capture.start_stream(audio_processor.process_audio)
        input("Press Enter to stop recording...\n")  # Keep running until user stops
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        audio_capture.stop_stream()
