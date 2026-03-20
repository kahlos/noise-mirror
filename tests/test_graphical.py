import os
import cv2
import numpy as np
from app.ai_engine import AIEngine
from app.prompt_manager import PromptManager
from app.cv_engine import extract_noise
from app.layout import build_4_quadrant_grid

def test_graphical_output():
    """
    Loads tests/output/test-photo.jpg, processes it through the entire layout and real AI pipeline,
    and saves the resulting 1024x1024 output interface grid to output_grid.jpg.
    """
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    input_path = os.path.join(output_dir, "test-photo.jpg")
    output_path = os.path.join(output_dir, "test_output_grid.jpg")
    
    assert os.path.exists(input_path), f"Test photo not found: {input_path}"
    
    # 1. Load and prepare frame
    frame = cv2.imread(input_path)
    cam_raw = cv2.resize(frame, (512, 512))
    cam_noise = extract_noise(cam_raw)
    
    # 2. Start AI Engine
    # Note: This will load the actual models in diffusers and take some seconds
    print("Starting AIEngine with real models...")
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    engine = AIEngine(models_dir=models_dir)
    
    pm = PromptManager()
    pm.set_manual_override("A beautiful masterpiece, detailed, 8k resolution")
    engine.start(pm)
    
    # 3. Submit frame to queue
    print("Submitting frame to AI queue...")
    engine.process_frame(cam_raw)
    
    # 4. Wait for AI inference (timeout 300 seconds to allow for downloads)
    print("Waiting for AI Inference to complete (this may take a while)...")
    ai_result = engine.get_latest_result(timeout=300.0)
    
    engine.stop()
    
    assert ai_result is not None, "AI Inference timed out or failed"
    ai_raw, ai_noise = ai_result
    
    # 5. Render Grid Layout
    grid = build_4_quadrant_grid(cam_raw, ai_raw, cam_noise, ai_noise)
    
    # 6. Apply UI Overlays
    display_frame = grid.copy()
    status_text = f"Prompt: {pm.get_current_prompt()[:50]}..."
    if pm.manual_prompt:
        status_text = "[MANUAL] " + status_text
        
    cv2.putText(display_frame, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
    # 7. Save output
    cv2.imwrite(output_path, display_frame)
    print(f"Graphical output successfully saved to: {output_path}")
    
    assert os.path.exists(output_path), "Failed to save output grid"

if __name__ == "__main__":
    test_graphical_output()
