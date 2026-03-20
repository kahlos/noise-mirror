import os
import cv2
import time
import numpy as np

from app.cv_engine import extract_noise
from app.layout import build_4_quadrant_grid
from app.prompt_manager import PromptManager
from app.ai_engine import AIEngine

class NoiseMirrorExhibit:
    def __init__(self):
        self.prompt_manager = PromptManager(rotation_interval=60.0)
        self.ai_engine = AIEngine(models_dir=os.path.join(os.getcwd(), "models"))
        self.running = False
        self.last_ai_result = None

    def run(self, camera_id=0):
        self.running = True
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
            
        print("Starting AI Engine...")
        self.ai_engine.start(self.prompt_manager)

        win_name = "The Noise Mirror - Museum Exhibit"
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

        print("Exhibit Started. Press 'TAB' to override prompt, 'Q' to quit.")

        # Dummy blank AI frames until first inference completes
        blank_frame = np.zeros((512, 512, 3), dtype=np.uint8)
        self.last_ai_result = (blank_frame, blank_frame)

        while self.running:
            # 1. Update Prompt Manager
            if self.prompt_manager.update():
                print(f"Auto-rotating to: {self.prompt_manager.get_current_prompt()}")

            # 2. Fetch Webcam (Fast Loop: 30FPS)
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            # Resize webcam frame for processing
            cam_raw = cv2.resize(frame, (512, 512))
            
            # OpenCV Math (Camera Noise Extract)
            cam_noise = extract_noise(cam_raw)
            
            # Push latest frame to Slow Loop (Drops stale frames)
            self.ai_engine.process_frame(cam_raw)
            
            # State Fetch (Non-blocking, grab latest AI result)
            ai_result = self.ai_engine.get_latest_result(timeout=None)
            if ai_result is not None:
                self.last_ai_result = ai_result
                
            ai_raw, ai_noise = self.last_ai_result
            
            # Render Grid
            grid = build_4_quadrant_grid(cam_raw, ai_raw, cam_noise, ai_noise)
            display_frame = grid

            # 3. Handle Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
            elif key == 9: # Tab key
                self.prompt_manager.pause_rotation()
                # Overlay instruction
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 450), (1024, 550), (0,0,0), -1)
                cv2.putText(overlay, "OVERRIDE ACTIVE: Enter new prompt in terminal...", 
                            (50, 512), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.imshow(win_name, overlay)
                cv2.waitKey(1)
                
                print("\n--- MANUAL PROMPT OVERRIDE ---")
                new_prompt = input("Enter prompt: ")
                if new_prompt.strip():
                    self.prompt_manager.set_manual_override(new_prompt)
                    print(f"Set prompt to: {new_prompt}")
                else:
                    self.prompt_manager.resume_auto_rotation()
                    print("Resumed auto-rotation.")
            
            # Overlay status
            current_p = self.prompt_manager.get_current_prompt()
            status_text = f"Prompt: {current_p[:50]}..."
            if self.prompt_manager.manual_prompt: status_text = "[MANUAL] " + status_text
            
            cv2.putText(display_frame, status_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(win_name, display_frame)

        print("Shutting down exhibit...")
        cap.release()
        self.ai_engine.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    exhibit = NoiseMirrorExhibit()
    exhibit.run()
