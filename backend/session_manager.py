# session_manager.py
import asyncio
import traceback
import time
from enum import Enum
from gemini_client import GeminiClient
from audio import AudioHandler
from video import VideoHandler
from tools.registry import ToolRegistry
from google.genai import types


class SessionState(Enum):
    """Session states for better control flow"""
    IDLE = "idle"
    USER_SPEAKING = "user_speaking"
    MODEL_RESPONDING = "model_responding"
    INTERRUPTED = "interrupted"


class SessionManager:
    def __init__(self, mode="none"):
        self.tool_registry = ToolRegistry()
        extra_tools = self.tool_registry.get_gemini_tool_declarations()
        self.gemini = GeminiClient(extra_tools=extra_tools)
        self.audio = AudioHandler()
        self.video = VideoHandler(mode)
        self.session = None
        
        # Session state management
        self.state = SessionState.IDLE
        self.state_lock = asyncio.Lock()
        
        # Flags to control data sources
        self.use_frontend_video = True
        self.use_frontend_audio = True
        
        # Optimized queue sizes with proper bounds
        self.audio_out_queue = asyncio.Queue(maxsize=10)  # User audio → Gemini
        self.audio_in_queue = asyncio.Queue(maxsize=10)   # Gemini audio → Speaker
        self.video_out_queue = asyncio.Queue(maxsize=3)   # Video frames → Gemini
        
        # Assign queues to handlers
        self.audio.audio_in_queue = self.audio_in_queue
        self.audio.out_queue = self.audio_out_queue
        self.video.out_queue = self.video_out_queue
        
        # Interruption handling
        self.user_interrupted = asyncio.Event()
        self.model_generating = asyncio.Event()
        
        # Performance monitoring
        self.audio_packets_sent = 0
        self.audio_packets_received = 0
        self.video_frames_sent = 0
        
        # Turn management for screen mode
        self.user_turn_active = asyncio.Event()
        self.last_user_activity = time.time()
        
    async def run(self):
        """Main session loop with proper error handling"""
        try:
            async with (
                self.gemini.connect() as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                
                # Core tasks
                tg.create_task(self._send_audio_stream())
                tg.create_task(self._send_video_stream())
                tg.create_task(self._receive_from_gemini())
                tg.create_task(self._monitor_interruptions())
                
                # Start local capture only if not using frontend
                if not self.use_frontend_audio:
                    tg.create_task(self.audio.listen_audio())
                
                if not self.use_frontend_video:
                    if self.video.video_mode == "camera":
                        tg.create_task(self.video.get_frames())
                    elif self.video.video_mode == "screen":
                        tg.create_task(self.video.get_screen())
                
                # Audio playback - CRITICAL: Uncomment for production
                tg.create_task(self.audio.play_audio())
                
                # Performance monitoring
                tg.create_task(self._log_performance())
                
                # Keep running
                await asyncio.Event().wait()
                
        except asyncio.CancelledError:
            print("✓ Session gracefully cancelled")
        except ExceptionGroup as eg:
            print("✗ Session error - ExceptionGroup:")
            traceback.print_exception(eg)
        except Exception as e:
            print(f"✗ Session error: {e}")
            traceback.print_exc()
        finally:
            await self._cleanup()
    
    async def _send_audio_stream(self):
        """High-priority audio sender with VAD support"""
        print("▶ Audio sender started")
        consecutive_empty = 0
        
        while True:
            try:
                # Wait for audio with timeout to check interruption
                try:
                    msg = await asyncio.wait_for(
                        self.audio_out_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    consecutive_empty += 1
                    if consecutive_empty > 50:  # 5 seconds of silence
                        # Send end_of_turn if user was speaking
                        async with self.state_lock:
                            if self.state == SessionState.USER_SPEAKING:
                                await self.session.send(input="", end_of_turn=True)
                                self.state = SessionState.IDLE
                                consecutive_empty = 0
                                print("⏸ User turn ended (silence)")
                    continue
                
                consecutive_empty = 0
                
                # Update state to user speaking
                async with self.state_lock:
                    if self.state != SessionState.USER_SPEAKING:
                        self.state = SessionState.USER_SPEAKING
                        self.user_turn_active.set()
                        # Interrupt model if it's speaking
                        if self.model_generating.is_set():
                            await self._handle_user_interruption()
                
                # Send audio to Gemini
                await self.session.send(input=msg)
                self.audio_packets_sent += 1
                self.last_user_activity = time.time()
                
                # Periodic logging
                if self.audio_packets_sent % 50 == 0:
                    print(f"→ Sent {self.audio_packets_sent} audio packets")
                    
            except Exception as e:
                print(f"✗ Error sending audio: {e}")
                await asyncio.sleep(0.01)
    
    async def _send_video_stream(self):
        """Background video sender with turn-taking for screen mode"""
        print("▶ Video sender started")
        
        while True:
            try:
                # For screen mode, only send when user is actively asking
                if self.video.video_mode == "screen":
                    # Wait for user to be speaking or just spoke
                    if not self.user_turn_active.is_set():
                        await asyncio.sleep(0.5)
                        continue
                    
                    # Don't send too frequently in screen mode
                    await asyncio.sleep(2.0)  # Send every 2 seconds max
                
                # Get frame with timeout
                try:
                    frame = await asyncio.wait_for(
                        self.video_out_queue.get(),
                        timeout=0.5
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Send frame
                await self.session.send(input=frame)
                self.video_frames_sent += 1
                
                if self.video_frames_sent % 10 == 0:
                    print(f"→ Sent {self.video_frames_sent} video frames")
                    
            except Exception as e:
                print(f"✗ Error sending video: {e}")
                await asyncio.sleep(0.1)
    
    async def _receive_from_gemini(self):
        """Receive responses with proper interruption handling"""
        print("▶ Gemini receiver started")
        
        while True:
            try:
                turn = self.session.receive()
                
                async for response in turn:
                    # Handle interruption signal from Gemini
                    if hasattr(response, 'server_content') and response.server_content:
                        if hasattr(response.server_content, 'interrupted') and response.server_content.interrupted:
                            print("🛑 Gemini detected interruption")
                            await self._clear_playback_queue()
                            async with self.state_lock:
                                self.state = SessionState.INTERRUPTED
                            continue
                    
                    # Handle audio data
                    if data := response.data:
                        # Set model generating flag
                        if not self.model_generating.is_set():
                            self.model_generating.set()
                            async with self.state_lock:
                                if self.state == SessionState.IDLE:
                                    self.state = SessionState.MODEL_RESPONDING
                        
                        # Add to playback queue with overflow protection
                        try:
                            self.audio_in_queue.put_nowait(data)
                            self.audio_packets_received += 1
                        except asyncio.QueueFull:
                            # Drop oldest packet if queue is full
                            try:
                                self.audio_in_queue.get_nowait()
                                self.audio_in_queue.put_nowait(data)
                            except:
                                pass
                        continue
                    
                    # Handle text responses
                    if text := response.text:
                        print(f"← Gemini: {text}")
                    
                    # Handle tool/function calls
                    tool_call = getattr(response, 'tool_call', None)
                    if tool_call:
                        for function_call in tool_call.function_calls:
                            print(f"🔧 Tool call: {function_call.name}({dict(function_call.args)})")
                            result = await self.tool_registry.execute(
                                function_call.name,
                                dict(function_call.args)
                            )
                            print(f"🔧 Tool result: {result}")
                            await self.session.send(
                                input=types.LiveClientToolResponse(
                                    function_responses=[
                                        types.FunctionResponse(
                                            name=function_call.name,
                                            id=function_call.id,
                                            response={"result": result},
                                        )
                                    ]
                                )
                            )
                
                # Turn complete
                print("✓ Turn complete")
                self.model_generating.clear()
                self.user_turn_active.clear()
                
                async with self.state_lock:
                    self.state = SessionState.IDLE
                # NOTE: Do NOT clear audio queue here — that would cut off Gemini's response.
                # Clearing only happens on actual interruption (_handle_user_interruption).
                
            except Exception as e:
                print(f"✗ Error receiving from Gemini: {e}")
                await asyncio.sleep(0.1)
    
    async def _monitor_interruptions(self):
        """Monitor for user interruptions during model response"""
        print("▶ Interruption monitor started")
        
        while True:
            try:
                await asyncio.sleep(0.05)  # Check every 50ms
                
                # If user starts speaking while model is generating
                if self.model_generating.is_set():
                    # Check if new audio is coming in
                    if not self.audio_out_queue.empty():
                        # User is speaking during model response
                        time_since_activity = time.time() - self.last_user_activity
                        if time_since_activity < 0.5:  # Recent user activity
                            print("🎤 User interruption detected")
                            await self._handle_user_interruption()
                            
            except Exception as e:
                print(f"✗ Interruption monitor error: {e}")
                await asyncio.sleep(0.1)
    
    async def _handle_user_interruption(self):
        """Handle user interrupting the model"""
        print("⚡ Handling user interruption")
        
        # Set interruption flag
        self.user_interrupted.set()
        
        # Clear model generation flag
        self.model_generating.clear()
        
        # Clear playback queue immediately
        await self._clear_playback_queue()
        
        # Update state
        async with self.state_lock:
            self.state = SessionState.USER_SPEAKING
        
        # Clear the interruption flag after a short delay
        await asyncio.sleep(0.2)
        self.user_interrupted.clear()
    
    async def _clear_playback_queue(self):
        """Clear audio playback queue quickly"""
        cleared = 0
        while not self.audio_in_queue.empty():
            try:
                self.audio_in_queue.get_nowait()
                cleared += 1
            except:
                break
        
        if cleared > 0:
            print(f"🗑 Cleared {cleared} audio packets from playback queue")
    
    async def _log_performance(self):
        """Log performance metrics periodically"""
        while True:
            await asyncio.sleep(10)
            print(f"""
                📊 Performance Stats:
                Audio sent: {self.audio_packets_sent}
                Audio received: {self.audio_packets_received}
                Video sent: {self.video_frames_sent}
                State: {self.state.value}
                Queues: Audio Out={self.audio_out_queue.qsize()}, 
                Audio In={self.audio_in_queue.qsize()}, 
                Video={self.video_out_queue.qsize()}
            """)
    
    async def enqueue_audio(self, data: bytes):
        """Enqueue audio from frontend with VAD"""
        audio_packet = {
            "data": data,
            "mime_type": "audio/pcm"
        }
        
        try:
            self.audio_out_queue.put_nowait(audio_packet)
        except asyncio.QueueFull:
            # Drop oldest packet
            try:
                self.audio_out_queue.get_nowait()
                self.audio_out_queue.put_nowait(audio_packet)
                print("⚠ Audio out queue full, dropped oldest")
            except:
                pass
    
    async def enqueue_video(self, data: dict):
        """Enqueue video/screen frames from frontend"""
        if "mime_type" not in data or "data" not in data:
            print(f"⚠ Invalid video frame format")
            return
        
        try:
            self.video_out_queue.put_nowait(data)
        except asyncio.QueueFull:
            # Drop oldest frame for real-time performance
            try:
                self.video_out_queue.get_nowait()
                self.video_out_queue.put_nowait(data)
            except:
                print("⚠ Video queue blocked, skipping frame")
    
    async def _cleanup(self):
        """Cleanup resources on session end"""
        print("🧹 Cleaning up session resources...")
        
        # Close audio stream if exists
        if hasattr(self.audio, 'audio_stream') and self.audio.audio_stream:
            try:
                self.audio.audio_stream.close()
            except:
                pass
        
        # Clear all queues
        for queue in [self.audio_out_queue, self.audio_in_queue, self.video_out_queue]:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except:
                    break
        
        print("✓ Cleanup complete")


class TextHandler:
    """Handle text interactions (for testing)"""
    def __init__(self):
        pass
    
    async def send_text(self, session):
        while True:
            text = await asyncio.to_thread(input, "message > ")
            if text.lower() == "q":
                break
            await session.send(input=text or ".", end_of_turn=True)
