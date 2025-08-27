import os
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

class AudioCapture:
    def __init__(self, sample_rate=16000, chunk_size=1024, silence_threshold=0.01, silence_duration=2):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.temp_files = []  # List to store temporary file paths
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
        print("Recording started... Speak into your microphone.")
    
    def stop_stream(self):
        """Stops the audio stream."""
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            print("Recording stopped.")
        
    def save_audio(self, audio_data, segment_number):
        """Saves the audio data to a temporary file in 16-bit PCM format."""
        file_name = f"temp_audio_segment_{segment_number}.wav"
        
        # Normalize audio_data to int16 range (-32768 to 32767)
        audio_data_int16 = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
        
        write(file_name, self.sample_rate, audio_data_int16)  # Save as 16-bit PCM
        self.temp_files.append(file_name)
        print(f"Audio segment saved as {file_name}.")

    def play_saved_files(self):
        """Play all saved audio files."""
        if not self.temp_files:
            print("No saved audio segments to play.")
            return

        for file in self.temp_files:
            print(f"Playing {file}...")
            os.system(f"afplay {file}")

    def delete_saved_files(self):
        """Delete all saved audio files."""
        for file in self.temp_files:
            if os.path.exists(file):
                os.remove(file)
        self.temp_files = []
        print("All saved audio segments deleted.")

class AudioProcessor:
    def __init__(self, sample_rate=16000, chunk_size=1024, silence_duration=2, silence_threshold=0.01):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_duration = silence_duration
        self.silence_threshold = silence_threshold
        self.segment_number = 0
        self.current_audio = np.array([], dtype=np.float32)
        self.silence_count = 0

    def process_audio(self, indata, frames, time, status):
        """Processes the audio input from the stream callback."""
        if status:
            print(f"Stream status: {status}")
        audio_data = np.frombuffer(indata, dtype=np.float32)
        self.current_audio = np.concatenate((self.current_audio, audio_data))

        # Calculate RMS to detect silence
        rms = np.sqrt(np.mean(audio_data**2))
        #print(f"RMS: {rms}")  # Debug line to see RMS values

        if rms < self.silence_threshold:
            self.silence_count += 1
        else:
            self.silence_count = 0

        # If silence persists for the specified duration, save the audio segment
        if self.silence_count >= (self.silence_duration * self.sample_rate / self.chunk_size):
            if len(self.current_audio) > 0:
                audio_capture.save_audio(self.current_audio, self.segment_number)
                self.segment_number += 1
                self.current_audio = np.array([], dtype=np.float32)  # Clear after saving
            self.silence_count = 0

# Main execution
if __name__ == "__main__":
    sample_rate = 16000
    chunk_size = 1024
    silence_duration = 2
    silence_threshold = 0.01

    audio_processor = AudioProcessor(sample_rate=sample_rate, chunk_size=chunk_size, silence_duration=silence_duration, silence_threshold=silence_threshold)
    audio_capture = AudioCapture(sample_rate=sample_rate, chunk_size=chunk_size, silence_threshold=silence_threshold, silence_duration=silence_duration)

    try:
        audio_capture.start_stream(audio_processor.process_audio)
        input("Press Enter to stop recording...\n")  # Keep running until user stops
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        audio_capture.stop_stream()

        while True:
            action = input("Press 'p' to play all saved audio files, 'd' to delete them, or 'q' to quit: ").lower()
            if action == 'p':
                audio_capture.play_saved_files()
            elif action == 'd':
                audio_capture.delete_saved_files()
            elif action == 'q':
                break
            else:
                print("Invalid input. Please press 'p', 'd', or 'q'.")