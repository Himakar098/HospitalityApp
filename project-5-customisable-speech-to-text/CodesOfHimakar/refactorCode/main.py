import logging
import numpy as np
from preprocess import AudioCapture, AudioProcessor
from model import SpeechToTextTranscriber

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("transcription.log"),
        logging.StreamHandler()
    ]
)

def audio_callback(indata, frames, time, status):
    """Callback function to capture audio chunks."""
    if status:
        logging.warning(f"Audio status: {status}")
    audio_data = indata[:, 0]
    audio_capture.current_audio = np.concatenate((audio_capture.current_audio, audio_data))

if __name__ == "__main__":
    silence_duration = 2

    # Initialize audio capture
    audio_capture = AudioCapture(silence_duration=silence_duration)
    audio_processor = AudioProcessor(silence_duration=silence_duration)
    transcriber = SpeechToTextTranscriber()
    audio_capture.transcriber = transcriber  # Inject transcriber

    try:
        # Start audio stream
        audio_capture.start_stream(audio_callback)
        
        # Simulate audio processing and transcription
        file_name = "segment.wav"
        audio_capture.save_and_transcribe_audio(audio_capture.current_audio, audio_capture.segment_number)
        transcription = transcriber.transcribe(file_name)
        logging.info(f"Transcription: {transcription}")

    except Exception as e:
        logging.error(f"Error occurred: {e}")
    finally:
        audio_capture.stop_stream()
        #audio_capture.cleanup()

