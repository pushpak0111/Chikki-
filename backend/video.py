# video.py
import asyncio, io, base64
import cv2, mss
import PIL.Image
import traceback

class VideoHandler:
    def __init__(self, mode="none"):
        self.video_mode = mode
        self.out_queue = None

    def _get_frame(self, cap):
        """Capture frame from camera"""
        try:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read camera frame")
                return None
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(frame_rgb)
            img.thumbnail([1024, 1024])

            image_io = io.BytesIO()
            img.save(image_io, format="jpeg", quality=85)
            image_io.seek(0)

            image_bytes = image_io.read()
            return {
                "mime_type": "image/jpeg", 
                "data": base64.b64encode(image_bytes).decode()
            }
        except Exception as e:
            print(f"❌ Error getting camera frame: {e}")
            return None
    
    async def get_frames(self):
        """Capture frames from camera"""
        try:
            print("📹 Initializing camera...")
            cap = await asyncio.to_thread(cv2.VideoCapture, 0)
            
            if not cap.isOpened():
                print("❌ Could not open camera")
                return
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("📹 Camera opened successfully")
            frame_count = 0

            while True:
                # Control frame rate FIRST
                await asyncio.sleep(1.0)  # 1 FPS
                
                frame = await asyncio.to_thread(self._get_frame, cap)
                if frame is None:
                    print("⚠️ Skipping null frame")
                    continue

                # Clear queue to keep only latest frame
                while not self.out_queue.empty():
                    try:
                        self.out_queue.get_nowait()
                    except:
                        break
                
                if self.out_queue:
                    try:
                        await self.out_queue.put(frame)
                        frame_count += 1
                        if frame_count == 1:
                            print("📹 First camera frame queued!")
                        if frame_count % 10 == 0:
                            print(f"📹 Camera: {frame_count} frames queued")
                    except Exception as e:
                        print(f"❌ Error queueing camera frame: {e}")

        except Exception as e:
            print(f"❌ Error in get_frames: {e}")
            traceback.print_exc()
        finally:
            if cap:
                cap.release()
                print("📹 Camera released")

    def _get_screen(self):
        """Capture screen frame"""
        try:
            sct = mss.mss()
            
            # FIXED: Use monitor[1] for primary display instead of [0]
            # monitor[0] is the combined virtual display of all monitors
            monitor = sct.monitors[1]  # Primary monitor
            
            # Capture screen
            screenshot = sct.grab(monitor)
            
            # Convert to PIL Image
            img = PIL.Image.frombytes(
                'RGB', 
                (screenshot.width, screenshot.height), 
                screenshot.rgb
            )
            
            # Resize for efficiency
            img.thumbnail([1280, 720])
            
            # Convert to JPEG
            image_io = io.BytesIO()
            img.save(image_io, format="jpeg", quality=75)
            image_io.seek(0)
            
            image_bytes = image_io.read()
            
            return {
                "mime_type": "image/jpeg", 
                "data": base64.b64encode(image_bytes).decode()
            }
            
        except Exception as e:
            print(f"❌ Error capturing screen: {e}")
            traceback.print_exc()
            return None

    async def get_screen(self):
        """Capture screen frames continuously"""
        try:
            print("🖥️ Starting screen capture...")
            
            # Test capture first
            test_frame = await asyncio.to_thread(self._get_screen)
            if test_frame is None:
                print("❌ Screen capture test failed")
                return
            
            print("🖥️ Screen capture initialized successfully")
            frame_count = 0

            while True:
                # Control frame rate FIRST
                await asyncio.sleep(1.0)  # 1 FPS for screen
                
                frame = await asyncio.to_thread(self._get_screen)
                if frame is None:
                    print("⚠️ Skipping null screen frame")
                    continue

                # Clear queue to keep only latest frame
                while not self.out_queue.empty():
                    try:
                        self.out_queue.get_nowait()
                    except:
                        break
                
                if self.out_queue:
                    try:
                        await self.out_queue.put(frame)
                        frame_count += 1
                        if frame_count == 1:
                            print("🖥️ First screen frame queued!")
                        if frame_count % 5 == 0:
                            print(f"🖥️ Screen: {frame_count} frames queued (queue size: {self.out_queue.qsize()})")
                    except Exception as e:
                        print(f"❌ Error queueing screen frame: {e}")

        except Exception as e:
            print(f"❌ Error in get_screen: {e}")
            traceback.print_exc()