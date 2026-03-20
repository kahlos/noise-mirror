
import os
import sys

# Set environment variables BEFORE any other imports
os.environ["TMPDIR"] = os.path.join(os.getcwd(), ".tmp")
os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".cache", "huggingface")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

import numpy as np
import cv2
import torch
import coremltools as ct
import time
from unittest.mock import MagicMock, patch

# MONKEY-PATCH for Headless IDE Environment
original_MLModel = ct.models.MLModel
def patched_MLModel(path, compute_units=None, **kwargs):
    print(f"  [Headless Patch] Loading {os.path.basename(path)} with CPU_ONLY")
    return original_MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY, **kwargs)
ct.models.MLModel = patched_MLModel

# Ensure local app is in path
sys.path.append(os.getcwd())

from app.main_exhibit import NoiseMirrorExhibit

class MockVideoCapture:
    def __init__(self, *args, **kwargs):
        self.is_opened = True
        self.frame_count = 0
        self.source_path = os.path.join(os.getcwd(), "test-photo.jpg")
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Test image not found at {self.source_path}")
        
        self.real_frame = cv2.imread(self.source_path)
        if self.real_frame is None:
             raise ValueError(f"Failed to load image from {self.source_path}")
        self.real_frame = cv2.resize(self.real_frame, (512, 512))
        
    def isOpened(self):
        return self.is_opened
        
    def read(self):
        self.frame_count += 1
        time.sleep(0.01) 
        return True, self.real_frame.copy()
        
    def release(self):
        self.is_opened = False

class SmartMockPipeline:
    def __init__(self, **kwargs):
        self.output_size = 512
        self.prompt = kwargs.get("prompt", "")
        self._current_prompt = self.prompt
        self._target_embeds = None 
        
    def process_frame(self, frame_bgr):
        out = cv2.resize(frame_bgr, (self.output_size, self.output_size))
        prompt_lower = self._current_prompt.lower()
        overlay = out.copy()
        
        if "red" in prompt_lower or "fire" in prompt_lower:
            overlay[:] = (0, 0, 255) # Red BGR
        elif "green" in prompt_lower or "nature" in prompt_lower:
            overlay[:] = (0, 255, 0) # Green BGR
        elif "blue" in prompt_lower or "water" in prompt_lower:
            overlay[:] = (255, 0, 0) # Blue BGR
        else:
            overlay[:] = (100, 200, 200) # Sepia/Default
            
        return cv2.addWeighted(out, 0.7, overlay, 0.3, 0)
        
    def to(self, device):
        return self


# Remove pytest decorator and use a simple loop
def run_test_prompt(mock_cv2_handles, prompt, expected_color):
    print(f"\n--- Testing Prompt: '{prompt}' ---")
    
    captured_frames = []
    def capture_frame(win_name, frame):
        captured_frames.append(frame.copy())
    mock_cv2_handles["imshow"].side_effect = capture_frame
    
    # Run for 5 frames then quit
    mock_cv2_handles["waitKey"].side_effect = [-1] * 4 + [ord('q')]
    
    with patch("app.main_exhibit.Pipeline") as MockPipelineClass:
        smart_mock = SmartMockPipeline(prompt=prompt)
        MockPipelineClass.return_value = smart_mock
        
        exhibit = NoiseMirrorExhibit()
        exhibit.prompt_manager.get_current_prompt = MagicMock(return_value=prompt)
        
        exhibit.run(camera_id=0)
    
    if len(captured_frames) == 0:
        print(f"FAILED: No frames captured for prompt: {prompt}")
        return False

    last_frame = captured_frames[-1]
    
    ai_quadrant = last_frame[0:512, 512:1024]
    mean_b, mean_g, mean_r = cv2.mean(ai_quadrant)[:3]
    print(f"Stats: R={mean_r:.2f}, G={mean_g:.2f}, B={mean_b:.2f}")
    
    success = False
    if expected_color == "green":
        success = (mean_g > mean_r and mean_g > mean_b)
    elif expected_color == "red":
        success = (mean_r > mean_g and mean_r > mean_b)
    elif expected_color == "blue":
        success = (mean_b > mean_g and mean_b > mean_r)
    elif expected_color == "sepia":
        success = (mean_g > 100 and mean_r > 100)
        
    if not success:
        print(f"FAILED: Color verification failed for prompt: {prompt}")
    else:
        print(f"SUCCESS: Color verification passed.")

    # Save artifact
    safe_prompt = prompt.lower().replace(" ", "_").replace(",", "")[:30]
    output_path = f"test_output_{safe_prompt}.jpg"
    cv2.imwrite(output_path, last_frame)
    print(f"Saved: {output_path}")
    return success

def main():
    test_cases = [
        ("A lush green forest, nature photography", "green"),
        ("A roaring fire in the red woods", "red"),
        ("Deep blue ocean waves", "blue"),
        ("A vintage sepia photograph", "sepia")
    ]
    
    with patch("cv2.VideoCapture", side_effect=MockVideoCapture), \
         patch("cv2.namedWindow"), \
         patch("cv2.imshow") as mock_imshow, \
         patch("cv2.waitKey") as mock_wait_key, \
         patch("cv2.destroyAllWindows"):
        
        mock_cv2_handles = {"imshow": mock_imshow, "waitKey": mock_wait_key}
        
        results = []
        for prompt, expected_color in test_cases:
            res = run_test_prompt(mock_cv2_handles, prompt, expected_color)
            results.append(res)
            
        if all(results):
            print("\n✅ ALL TESTS PASSED")
            sys.exit(0)
        else:
            print("\n❌ SOME TESTS FAILED")
            sys.exit(1)

if __name__ == "__main__":
    main()
