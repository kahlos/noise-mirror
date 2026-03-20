import cv2
import time
import queue
import threading
import base64
import requests
import numpy as np
from app.cv_engine import extract_noise

class AIEngine:
    def __init__(self, api_url="http://127.0.0.1:7860/sdapi/v1/img2img"):
        self.api_url = api_url
        
        # Performance: Reuse TCP connections to API
        self.session = requests.Session()
        
        # Thread communication
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.running = False
        self.prompt_manager = None
        self.thread = None

    def start(self, prompt_manager):
        self.running = True
        self.prompt_manager = prompt_manager
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

    def process_frame(self, frame):
        """Put frame into the Slow Loop queue (non-blocking, dropping stale)."""
        if not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(frame)

    def get_latest_result(self, timeout=None):
        """Retrieve the latest completely rendered AI frame & noise."""
        if timeout is None:
            if self.result_queue.empty():
                return None
            try:
                return self.result_queue.get_nowait()
            except queue.Empty:
                return None
        else:
            try:
                return self.result_queue.get(timeout=timeout)
            except queue.Empty:
                return None

    def _inference_loop(self):
        print("AI Inference Slow Loop (Draw Things API) started.")
        while self.running:
            loop_start = time.time()
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if not self.prompt_manager:
                continue

            current_prompt = self.prompt_manager.get_current_prompt()
            
            # Pre-process: Resize and encode frame to Base64 for the API
            frame_resized = cv2.resize(frame, (448, 448))
            _, buffer = cv2.imencode('.png', frame_resized)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Construct the payload for Draw Things (A1111 compatible)
            payload = {
                "init_images": [frame_b64],
                "prompt": current_prompt,
                "steps": 2,                # Adjust based on your Draw Things model
                "width": 448,
                "height": 448,
                "denoising_strength": 0.5  # Ensure the image noise is blended
            }
            
            try:
                # Send to Draw Things
                response = self.session.post(self.api_url, json=payload, timeout=10.0)
                response.raise_for_status()
                response_data = response.json()
                
                # Extract the returned Base64 image
                result_b64 = response_data['images'][0]
                
                # Draw Things / A1111 sometimes prepends a data URI scheme
                if result_b64.startswith("data:image"):
                    result_b64 = result_b64.split(",", 1)[1]
                    
                # Decode Base64 back to an OpenCV BGR frame
                img_bytes = base64.b64decode(result_b64)
                img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                ai_raw = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                
                # Safety check in case decoding fails
                if ai_raw is None:
                    continue
                    
                # Resize from 448x448 back to 512x512 for main UI layout
                ai_raw = cv2.resize(ai_raw, (512, 512))
                
                # Thermal cap: ensure we don't exceed 3 FPS (0.33s per frame)
                elapsed = time.time() - loop_start
                if elapsed < 0.333:
                    time.sleep(0.333 - elapsed)
                
                # Post-process (extract noise from the AI generated frame)
                ai_noise = extract_noise(ai_raw)
                
                # Publish to State Fetch Queue
                if not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.result_queue.put((ai_raw, ai_noise))
                
            except requests.exceptions.RequestException as e:
                print(f"API Connection Error (Is Draw Things running?): {e}")
                time.sleep(2.0) # Back off to prevent console spam
            except Exception as e:
                print(f"Inference processing error: {e}")