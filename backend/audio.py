# audio.py
import asyncio
import pyaudio
import numpy as np
from collections import deque
import time

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    print("⚠ webrtcvad not installed. Install with: pip install webrtcvad-wheels")


# Audio Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 512

# VAD Configuration
VAD_MODE = 1  # 0-3, where 3 is most aggressive
FRAME_DURATION_MS = 30  # Must be 10, 20, or 30 ms
SILENCE_THRESHOLD = 1  # Seconds of silence before end of speech


class JitterBuffer:
    """Jitter buffer for smooth audio playback"""
    def __init__(self, target_size=5, max_size=10):
        self.buffer = deque(maxlen=max_size)
        self.target_size = target_size
        self.underrun_count = 0
        self.overrun_count = 0
    
    def add(self, data):
        """Add audio chunk to buffer"""
        try:
            self.buffer.append(data)
            if len(self.buffer) > self.target_size * 1.5:
                self.overrun_count += 1
        except:
            pass
    
    def get(self):
        """Get audio chunk from buffer"""
        if len(self.buffer) >= self.target_size:
            return self.buffer.popleft()
        elif len(self.buffer) > 0:
            # Buffer underrun, but we have some data
            self.underrun_count += 1
            return self.buffer.popleft()
        else:
            # Complete underrun - return silence
            self.underrun_count += 1
            return b'\x00\x00' * CHUNK_SIZE
    
    def is_ready(self):
        """Check if buffer has enough data for smooth playback"""
        return len(self.buffer) >= self.target_size
    
    def clear(self):
        """Clear all buffered audio"""
        self.buffer.clear()
    
    def size(self):
        return len(self.buffer)


class AudioHandler:
    def __init__(self):
        self.pya = pyaudio.PyAudio()
        self.audio_in_queue = None  # Will be set by SessionManager
        self.out_queue = None       # Will be set by SessionManager
        
        # Streams
        self.audio_stream = None
        self.playback_stream = None
        
        # VAD
        self.vad = None
        self.vad_enabled = VAD_AVAILABLE
        if self.vad_enabled:
            self.vad = webrtcvad.Vad(VAD_MODE)
            print(f"✓ WebRTC VAD initialized (mode={VAD_MODE})")
        
        # Speech detection state
        self.is_speech_active = False
        self.silence_start = None
        self.speech_frames = deque(maxlen=10)  # Keep last 10 frames for context
        
        # Jitter buffer for playback
        self.jitter_buffer = JitterBuffer(target_size=5, max_size=10)
        
        # Performance metrics
        self.frames_captured = 0
        self.frames_played = 0
        self.vad_speech_frames = 0
        self.vad_silence_frames = 0
    
    async def listen_audio(self):
        """Capture audio from microphone with VAD"""
        print("🎤 Starting microphone capture...")
        
        try:
            mic_info = self.pya.get_default_input_device_info()
            self.audio_stream = await asyncio.to_thread(
                self.pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                input_device_index=mic_info["index"],
                frames_per_buffer=CHUNK_SIZE,
            )
            
            print(f"✓ Microphone opened: {mic_info['name']}")
            
            kwargs = {"exception_on_overflow": False}
            consecutive_silence = 0
            
            while True:
                # Read audio chunk
                data = await asyncio.to_thread(
                    self.audio_stream.read,
                    CHUNK_SIZE,
                    **kwargs
                )
                
                self.frames_captured += 1
                
                # Apply VAD if enabled
                if self.vad_enabled:
                    is_speech = self._detect_speech(data)
                    
                    if is_speech:
                        # Speech detected
                        self.vad_speech_frames += 1
                        consecutive_silence = 0
                        
                        if not self.is_speech_active:
                            print("🗣 Speech started")
                            self.is_speech_active = True
                        
                        # Send audio packet
                        await self.out_queue.put({
                            "data": data,
                            "mime_type": "audio/pcm"
                        })
                        
                    else:
                        # Silence detected
                        self.vad_silence_frames += 1
                        consecutive_silence += 1
                        
                        # If we were speaking, send a few more frames of context
                        if self.is_speech_active and consecutive_silence < 10:
                            await self.out_queue.put({
                                "data": data,
                                "mime_type": "audio/pcm"
                            })
                        
                        # End of speech detection
                        if consecutive_silence > int(SILENCE_THRESHOLD * SEND_SAMPLE_RATE / CHUNK_SIZE):
                            if self.is_speech_active:
                                print("⏸ Speech ended")
                                self.is_speech_active = False
                                consecutive_silence = 0
                else:
                    # No VAD - send everything
                    await self.out_queue.put({
                        "data": data,
                        "mime_type": "audio/pcm"
                    })
                
                # Periodic stats
                if self.frames_captured % 500 == 0:
                    speech_ratio = (self.vad_speech_frames / self.frames_captured * 100) if self.frames_captured > 0 else 0
                    print(f"🎤 Captured {self.frames_captured} frames | Speech: {speech_ratio:.1f}%")
                    
        except Exception as e:
            print(f"✗ Microphone capture error: {e}")
        finally:
            if self.audio_stream:
                self.audio_stream.close()
                print("✓ Microphone closed")
    
    def _detect_speech(self, audio_data):
        """Detect speech using WebRTC VAD"""
        if not self.vad_enabled or not self.vad:
            return True  # If no VAD, assume all audio is speech
        
        try:
            # VAD requires specific frame durations (10, 20, or 30 ms)
            # Calculate required frame size
            frame_size = int(SEND_SAMPLE_RATE * FRAME_DURATION_MS / 1000)
            
            # If audio_data is not the right size, pad or truncate
            if len(audio_data) < frame_size * 2:  # *2 because 16-bit = 2 bytes
                # Pad with zeros
                audio_data = audio_data + b'\x00' * (frame_size * 2 - len(audio_data))
            elif len(audio_data) > frame_size * 2:
                # Truncate
                audio_data = audio_data[:frame_size * 2]
            
            # Run VAD
            is_speech = self.vad.is_speech(audio_data, SEND_SAMPLE_RATE)
            
            # Store in history
            self.speech_frames.append(is_speech)
            
            # Use majority voting over recent frames to reduce jitter
            if len(self.speech_frames) >= 3:
                recent_speech = sum(self.speech_frames[-3:])
                return recent_speech >= 2  # At least 2 out of 3 frames are speech
            
            return is_speech
            
        except Exception as e:
            print(f"✗ VAD error: {e}")
            return True  # On error, assume speech
    
    async def play_audio(self):
        """Play audio from Gemini with jitter buffer"""
        print("🔊 Starting audio playback...")
        
        try:
            self.playback_stream = await asyncio.to_thread(
                self.pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE,
                output=True,
            )
            
            print("✓ Playback stream opened")
            
            # Wait for buffer to fill initially
            while not self.jitter_buffer.is_ready():
                if not self.audio_in_queue.empty():
                    data = await self.audio_in_queue.get()
                    self.jitter_buffer.add(data)
                else:
                    await asyncio.sleep(0.01)
            
            print("✓ Jitter buffer ready")
            
            while True:
                # Fill jitter buffer from queue
                while not self.audio_in_queue.empty() and self.jitter_buffer.size() < 10:
                    try:
                        data = self.audio_in_queue.get_nowait()
                        self.jitter_buffer.add(data)
                    except:
                        break
                
                # Get audio from jitter buffer
                audio_chunk = self.jitter_buffer.get()
                
                # Play audio
                await asyncio.to_thread(self.playback_stream.write, audio_chunk)
                self.frames_played += 1
                
                # Periodic stats
                if self.frames_played % 500 == 0:
                    print(f"🔊 Played {self.frames_played} frames | Buffer: {self.jitter_buffer.size()} | Underruns: {self.jitter_buffer.underrun_count}")
                
                # Small delay to prevent tight loop
                await asyncio.sleep(0.001)
                
        except Exception as e:
            print(f"✗ Playback error: {e}")
        finally:
            if self.playback_stream:
                self.playback_stream.close()
                print("✓ Playback stream closed")
    
    def clear_playback_buffer(self):
        """Clear jitter buffer for interruptions"""
        self.jitter_buffer.clear()
        print("🗑 Playback buffer cleared")


# Legacy class for backward compatibility
class AudioHandlerOld:
    def __init__(self):
        self.pya = pyaudio.PyAudio()
        self.audio_out_queue = asyncio.Queue()
        self.audio_in_queue = asyncio.Queue()
    
    async def record_microphone(self):
        print("⚠ Using deprecated AudioHandlerOld. Please use AudioHandler instead.")
        mic_info = self.pya.get_default_input_device_info()
        stream = self.pya.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        while True:
            data = await asyncio.to_thread(stream.read, CHUNK_SIZE, exception_on_overflow=False)
            await self.audio_out_queue.put({"data": data, "mime_type": "audio/pcm"})
    
    async def play_audio(self):
        stream = self.pya.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        while True:
            try:
                bytestream = await self.audio_in_queue.get()
                if bytestream is None or bytestream == b"\x00" * len(bytestream):
                    break
                stream.write(bytestream)
            except Exception as e:
                print(f"Playback error: {e}")
