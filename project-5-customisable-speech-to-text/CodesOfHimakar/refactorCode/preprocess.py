import os
import logging
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from model import SpeechToTextTranscriber

class AudioCapture:
    def __init__(self, sample_rate=16000, chunk_size=1024, silence_threshold=0.01, silence_duration=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.temp_files = []  # List to store temporary file paths
        self.audio_stream = None
        self.transcriber = SpeechToTextTranscriber()  # Instantiate transcriber
        self.segment_number = 0
        self.current_audio = np.array([], dtype=np.float32)

    def start_stream(self, callback):
        """Starts the audio stream with the given callback for processing."""
        try:
            self.audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.chunk_size,
                callback=callback
            )
            self.audio_stream.start()
            logging.info("Recording started... Speak into your microphone.")
        except Exception as e:
            logging.error(f"Error starting audio stream: {e}")

    def stop_stream(self):
        """Stops the audio stream and ensures all transcriptions complete."""
        try:
            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
                logging.info("Recording stopped.")
                self.finish_transcriptions()
        except Exception as e:
            logging.error(f"Error stopping audio stream: {e}")

    def finish_transcriptions(self):
        """Waits for all transcription threads to complete."""
        logging.info("Waiting for all transcriptions to complete...")
        try:
            self.transcriber.wait_for_completion()
            logging.info("All transcriptions are complete.")
            self.play_or_delete_options()  # After transcription, offer options to the user
        except Exception as e:
            logging.error(f"Error finishing transcriptions: {e}")

    def save_and_transcribe_audio(self, audio_data, segment_number):
        """Saves the audio data to a temporary file and starts transcription in a separate thread."""
        if len(audio_data) == 0:
            logging.warning(f"Audio segment {segment_number} is empty, skipping save and transcription.")
            return  # Skip further processing if no audio data is captured

        try:
            file_name = f"temp_audio_segment_{segment_number}.wav"

            # Normalize audio_data to int16 range (-32768 to 32767)
            if np.max(np.abs(audio_data)) != 0:
                audio_data_int16 = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
            else:
                logging.warning("Audio segment contains silence only, skipping normalization and transcription.")
                return  # Skip processing this audio segment

            write(file_name, self.sample_rate, audio_data_int16)  # Save as 16-bit PCM
            self.temp_files.append(file_name)
            logging.info(f"Audio segment saved as {file_name}.")

            # Start a new thread to handle transcription
            self.transcriber.transcribe(file_name)
        except Exception as e:
            logging.error(f"Error saving or transcribing audio segment {segment_number}: {e}")

    def play_or_delete_options(self):
        """Provides options to play, delete, or re-transcribe saved audio files."""
        while True:
            action = input("Press 'p' to play all saved audio files, 'd' to delete them, 'r' to re-transcribe, or 'q' to quit: ").lower()
            if action == 'p':
                self.play_files()
            elif action == 'd':
                self.delete_files()
            elif action == 'r':
                self.retranscribe_files()
            elif action == 'q':
                logging.info("Exiting...")
                break
            else:
                logging.warning("Invalid input, please try again.")

    def play_files(self):
        """Plays all saved audio files."""
        for file in self.temp_files:
            try:
                logging.info(f"Playing {file}...")
                os.system(f"ffplay -nodisp -autoexit {file}")
            except Exception as e:
                logging.error(f"Error playing file {file}: {e}")

    def delete_files(self):
        """Deletes all saved audio files."""
        for file in self.temp_files:
            try:
                os.remove(file)
                logging.info(f"Deleted {file}.")
            except Exception as e:
                logging.error(f"Error deleting file {file}: {e}")
        self.temp_files.clear()

    def retranscribe_files(self):
        """Re-transcribes all saved audio files."""
        for file in self.temp_files:
            try:
                self.transcriber.transcribe(file)
            except Exception as e:
                logging.error(f"Error re-transcribing file {file}: {e}")

class AudioProcessor:
    def __init__(self, sample_rate=16000, chunk_size=1024, silence_duration=1, silence_threshold=0.01):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_duration = silence_duration
        self.silence_threshold = silence_threshold
        self.segment_number = 0
        self.current_audio = np.array([], dtype=np.float32)
        self.silence_count = 0
        self.MIN_AUDIO_LENGTH = self.sample_rate * 0.5  # 0.5 seconds
        self.MAX_AUDIO_LENGTH = self.sample_rate * 30  # 30 seconds
        self.SLIDING_WINDOW = self.sample_rate * 2  # 2 seconds

    def process_audio(self, indata, frames, time, status, audio_capture):
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

        # If silence persists for the specified duration, save the audio segment
        if self.silence_count >= (self.silence_duration * self.sample_rate / self.chunk_size):
            if len(self.current_audio) > self.MIN_AUDIO_LENGTH:
                audio_capture.save_and_transcribe_audio(self.current_audio, self.segment_number)
                self.segment_number += 1
                self.current_audio = np.array([], dtype=np.float32)  # Clear after saving
            self.silence_count = 0

        # If the audio exceeds 30 seconds, chunk it with a sliding window
        if len(self.current_audio) > self.MAX_AUDIO_LENGTH:
            logging.info("Chunking audio with sliding window due to exceeding length...")
            start = len(self.current_audio) - self.MAX_AUDIO_LENGTH
            end = len(self.current_audio) - self.SLIDING_WINDOW
            chunk = self.current_audio[start:end]
            audio_capture.save_and_transcribe_audio(chunk, self.segment_number)
            self.segment_number += 1
            self.current_audio = self.current_audio[-self.SLIDING_WINDOW:]  # Keep the last 2 seconds
