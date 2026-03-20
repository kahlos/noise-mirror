import os
import cv2
import time
import numpy as np
from app.ai_engine import AIEngine
from app.prompt_manager import PromptManager
from app.cv_engine import extract_noise
from app.layout import build_4_quadrant_grid

def test_video_proxy():
    """
    Simulates the main exhibit live loop, but using a pre-recorded video 
    (test-video.mov) as the input source instead of the webcam.
    This validates that the fast loop and background AI loop work concurrently.
    """
    video_path = os.path.join(os.path.dirname(__file__), "test-video.mov")
    output_path = os.path.join(os.path.dirname(__file__), "test_output_video.mp4")
    
    assert os.path.exists(video_path), f"Test video not found: {video_path}"
    
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video file: {video_path}"
    
    # Get video properties for output
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps: # Handle NaN or 0
        fps = 30.0
        
    # We expect our grid to be 1024x1024
    frame_width = 1024
    frame_height = 1024
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize Engine
    print("Starting AIEngine with real models...")
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    engine = AIEngine(models_dir=models_dir)
    
    pm = PromptManager()
    pm.set_manual_override("A cyberpunk city street, neon lights, masterpiece, 8k")
    engine.start(pm)
    
    print("Warming up AI engine (compiling shaders may take 1-2 minutes)...")
    ret, warmup_frame = cap.read()
    if ret:
        cam_raw = cv2.resize(warmup_frame, (512, 512))
        engine.process_frame(cam_raw)
        while True:
            res = engine.get_latest_result(timeout=1.0)
            if res is not None:
                last_ai_result = res
                successful_inferences = 1
                print("✅ AI Engine warmed up!")
                break
    else:
        # Fallback if video is empty
        blank_frame = np.zeros((512, 512, 3), dtype=np.uint8)
        last_ai_result = (blank_frame, blank_frame)
        successful_inferences = 0
        
    # Rewind video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_count = 0
    start_time = time.time()
    
    print("Processing video frames (simulating 30 FPS live feed)...")
    while cap.isOpened():
        loop_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize webcam frame for processing
        cam_raw = cv2.resize(frame, (512, 512))
        
        # OpenCV Math (Camera Noise Extract)
        cam_noise = extract_noise(cam_raw)
        
        # Push latest frame to Slow Loop
        engine.process_frame(cam_raw)
        
        # State Fetch (Non-blocking)
        ai_result = engine.get_latest_result(timeout=None)
        if ai_result is not None:
            last_ai_result = ai_result
            successful_inferences += 1
            print(f"✅ AI Inference {successful_inferences} completed!")
        ai_raw, ai_noise = last_ai_result
        
        # Render Grid
        grid = build_4_quadrant_grid(cam_raw, ai_raw, cam_noise, ai_noise)
        
        # Apply UI Overlays
        display_frame = grid.copy()
        status_text = f"Prompt: {pm.get_current_prompt()[:50]}... Frame: {frame_count}"
        if pm.manual_prompt:
            status_text = "[PROXY TEST] " + status_text
            
        cv2.putText(display_frame, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
        out.write(display_frame)
        frame_count += 1
        
        # Optional: yield to testing environment or debug logging
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
            
        # Simulate real-time webcam frame rate
        elapsed = time.time() - loop_start
        time_per_frame = 1.0 / fps
        if elapsed < time_per_frame:
            time.sleep(time_per_frame - elapsed)
            
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Finished processing {frame_count} frames in {duration:.2f} seconds.")
    
    cap.release()
    out.release()
    engine.stop()
    
    print(f"Video output successfully saved to: {output_path}")
    assert os.path.exists(output_path), "Failed to save output video"
    
if __name__ == "__main__":
    test_video_proxy()
