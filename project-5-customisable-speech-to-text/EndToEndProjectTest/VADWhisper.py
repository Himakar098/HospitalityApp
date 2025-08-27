import sounddevice as sd
import numpy as np
import torch
import threading
import queue
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class RealtimeVADWhisperTranscriber:
    def __init__(self, sample_rate=16000, channels=1, amplification_factor=2.0, language="telugu"):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.input_thread = threading.Thread(target=self.wait_for_enter)
        self.amplification_factor = amplification_factor
        self.language = language
        
        # Load Silero VAD
        self.vad_model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', 
                                                    model='silero_vad', 
                                                    force_reload=True)
        self.vad_model.eval()
        
        # Load Distil Whisper
        self.whisper_processor = WhisperProcessor.from_pretrained("distil-whisper/distil-large-v2")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("distil-whisper/distil-large-v2")
        self.whisper_model.config.forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(language=language, task="transcribe")
        
        # Set parameters
        self.window_size_samples = 512
        self.speech_threshold = 0.3
        self.transcription_buffer = []
        self.last_transcription_time = time.time()

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")
        amplified_indata = np.clip(indata * self.amplification_factor, -32768, 32767)
        self.audio_queue.put(amplified_indata.copy())

    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1/32768
        sound = sound.squeeze()
        return sound

    def process_audio(self, audio_chunk):
        audio_float = self.int2float(audio_chunk)
        audio_tensor = torch.tensor(audio_float).unsqueeze(0)
        
        if audio_tensor.shape[1] != self.window_size_samples:
            return 0
        
        try:
            with torch.no_grad():
                speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
        except Exception as e:
            print(f"Error during VAD inference: {e}")
            return 0
        
        return speech_prob

    def is_speech(self, prob):
        return prob > self.speech_threshold

    def transcribe_audio(self, audio_buffer):
        audio_array = np.concatenate(audio_buffer)
        input_features = self.whisper_processor(audio_array, sampling_rate=self.sample_rate, return_tensors="pt").input_features
        
        with torch.no_grad():
            predicted_ids = self.whisper_model.generate(input_features)
        
        transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0]

    def visualize_audio(self, audio_chunk, speech_prob):
        energy = np.sqrt(np.mean(audio_chunk**2))
        energy_bar = '#' * int(energy * 50)
        speech_indicator = 'SPEECH' if self.is_speech(speech_prob) else 'SILENCE'
        return f"[{energy_bar:<50}] {speech_indicator} | Prob: {speech_prob:.4f} | Energy: {energy:.4f}"

    def start_streaming(self):
        print(f"Starting real-time audio processing with VAD and Distil Whisper. Speak in {self.language}.")
        print("Press Enter to stop.")
        
        self.input_thread.start()
        
        with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, 
                            callback=self.audio_callback, blocksize=self.window_size_samples):
            while not self.stop_event.is_set():
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    speech_prob = self.process_audio(audio_chunk)
                    visualization = self.visualize_audio(audio_chunk, speech_prob)
                    
                    if self.is_speech(speech_prob):
                        self.transcription_buffer.append(audio_chunk)
                    
                    current_time = time.time()
                    if current_time - self.last_transcription_time > 2 and self.transcription_buffer:  # Transcribe every 2 seconds if we have speech
                        transcription = self.transcribe_audio(self.transcription_buffer)
                        print(f"\nTranscription: {transcription}")
                        self.transcription_buffer = []
                        self.last_transcription_time = current_time
                    
                    print(f"\r{visualization}", end='', flush=True)
                    
                except queue.Empty:
                    pass
                except Exception as e:
                    print(f"\nError processing audio: {e}")

    def wait_for_enter(self):
        input("Press Enter to stop...")
        self.stop_streaming()

    def stop_streaming(self):
        print("\nStopping audio stream.")
        self.stop_event.set()

if __name__ == "__main__":
    transcriber = RealtimeVADWhisperTranscriber(language="telugu")
    try:
        transcriber.start_streaming()
    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        transcriber.stop_streaming()