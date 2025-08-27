import sounddevice as sd
import numpy as np
import torch
import threading
import queue
import time

class DebugSileroVAD:
    def __init__(self, sample_rate=16000, channels=1, amplification_factor=2.0):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.input_thread = threading.Thread(target=self.wait_for_enter)
        self.amplification_factor = amplification_factor  # Amplification factor
        
        # Load Silero VAD
        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', 
                                                 model='silero_vad', 
                                                 force_reload=True)
        self.model.eval()
        
        # Set parameters
        self.window_size_samples = 512
        self.speech_threshold = 0.3  # Adjust threshold as needed

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")
        
        # Apply amplification
        amplified_indata = indata * self.amplification_factor
        
        # Ensure that values are within valid range for audio
        np.clip(amplified_indata, -32768, 32767, out=amplified_indata)
        
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
        audio_tensor = torch.tensor(audio_float).unsqueeze(0)  # Add batch dimension
        
        # Ensure correct shape for the model
        if len(audio_tensor.shape) != 2 or audio_tensor.shape[1] != self.window_size_samples:
            print(f"Audio tensor shape mismatch: {audio_tensor.shape}")
            return 0
        
        try:
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sample_rate).item()
        except Exception as e:
            print(f"Error during model inference: {e}")
            return 0
        
        print(f"Processed Audio Tensor: {audio_tensor}")
        print(f"Speech Probability: {speech_prob}")
        return speech_prob

    def is_speech(self, prob):
        return prob > self.speech_threshold

    def visualize_audio(self, audio_chunk, speech_prob):
        energy = np.sqrt(np.mean(audio_chunk**2))
        energy_bar = '#' * int(energy * 50)
        speech_indicator = 'SPEECH' if self.is_speech(speech_prob) else 'SILENCE'
        return f"[{energy_bar:<50}] {speech_indicator} | Prob: {speech_prob:.4f} | Energy: {energy:.4f}"

    def start_streaming(self):
        print("Starting real-time audio processing with Silero VAD. Speak into your microphone.")
        print("Press Enter to stop.")
        
        self.input_thread.start()  # Start the thread that waits for Enter key press
        
        with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, 
                            callback=self.audio_callback, blocksize=self.window_size_samples):
            while not self.stop_event.is_set():
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    
                    start_time = time.time()
                    
                    speech_prob = self.process_audio(audio_chunk)
                    visualization = self.visualize_audio(audio_chunk, speech_prob)
                    
                    processing_time = time.time() - start_time
                    
                    print(f"\r{visualization} | Prob: {speech_prob:.4f} | Time: {processing_time*1000:.2f} ms | Max: {np.max(np.abs(audio_chunk)):.4f}", end='', flush=True)
                    
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
    vad_processor = DebugSileroVAD()
    try:
        vad_processor.start_streaming()
    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        vad_processor.stop_streaming()
