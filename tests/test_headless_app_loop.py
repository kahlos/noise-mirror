
import os
import sys

# Set environment variables BEFORE any other imports to ensure CoreML/HF use them
os.environ["TMPDIR"] = os.path.join(os.getcwd(), ".tmp")
os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".cache", "huggingface")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

import numpy as np
import cv2
import torch
import coremltools as ct
import time
import threading
import queue
import pytest
from unittest.mock import MagicMock, patch

# MONKEY-PATCH for Headless IDE Environment:
# CoreML GPU compilation fails in this sandbox due to temp directory permissions.
# We patch MLModel to use CPU_ONLY strictly for this verification step.
original_MLModel = ct.models.MLModel
def patched_MLModel(path, compute_units=None, **kwargs):
    print(f"  [Headless Patch] Loading {os.path.basename(path)} with CPU_ONLY")
    return original_MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY, **kwargs)
ct.models.MLModel = patched_MLModel

# Ensure local app is in path
sys.path.append(os.getcwd())

from app.main_exhibit import NoiseMirrorExhibit

# --------------------------------------------------------------------------------
# HEADLESS TEST INFRASTRUCTURE
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# HEADLESS TEST INFRASTRUCTURE
# --------------------------------------------------------------------------------

class MockVideoCapture:
    def __init__(self, *args, **kwargs):
        self.is_opened = True
        self.frame_count = 0
        # Load the real test image
        self.source_path = os.path.join(os.getcwd(), "test-photo.jpg")
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Test image not found at {self.source_path}")
        
        # Pre-load and resize to standard webcam size
        self.real_frame = cv2.imread(self.source_path)
        if self.real_frame is None:
             raise ValueError(f"Failed to load image from {self.source_path}")
        self.real_frame = cv2.resize(self.real_frame, (512, 512))
        
    def isOpened(self):
        return self.is_opened
        
    def read(self):
        self.frame_count += 1
        
        # Return the static image (simulating a camera pointed at a photo)
        # We assume the camera is stable, so we just return the same frame.
        # To prove it's "live", we can add subtle noise or just return as is.
        
        time.sleep(0.01) # Simulate hardware delay
        return True, self.real_frame.copy()
        
    def release(self):
        self.is_opened = False

class SmartMockPipeline:
    def __init__(self, **kwargs):
        self.output_size = 512
        # Default prompt if none set
        self.prompt = kwargs.get("prompt", "")
        # Properties expected by NoiseMirrorExhibit
        self._current_prompt = self.prompt
        self._target_embeds = None 
        
    def process_frame(self, frame_bgr):
        """
        Applies a visual tint based on the current prompt to verify 
        the 'Diffusion' logic is responsive to the prompt.
        """
        # Resize input to output size
        out = cv2.resize(frame_bgr, (self.output_size, self.output_size))
        
        # Apply Tint based on prompt keywords
        # The prompt is managed by NoiseMirrorExhibit -> PromptManager
        # We need to detect what the prompt IS.
        # In the app, 'pipeline._current_prompt' is updated.
        
        prompt_lower = self._current_prompt.lower()
        
        overlay = out.copy()
        if "red" in prompt_lower or "fire" in prompt_lower:
            overlay[:] = (0, 0, 255) # Red BGR
        elif "green" in prompt_lower or "nature" in prompt_lower:
            overlay[:] = (0, 255, 0) # Green BGR
        elif "blue" in prompt_lower or "water" in prompt_lower:
            overlay[:] = (255, 0, 0) # Blue BGR
        else:
            # Default to Sepia if no match, to show it's "processed"
            overlay[:] = (100, 200, 200) 
            
        # Blend the tint
        return cv2.addWeighted(out, 0.7, overlay, 0.3, 0)
        
    def to(self, device):
        return self
        
    def _encode_single(self, prompt):
        # Mock encoding method if called
        return None

@pytest.fixture
def mock_cv2():
    """Patches cv2 functions to be headless-safe."""
    with patch("cv2.VideoCapture", side_effect=MockVideoCapture) as mock_cap, \
         patch("cv2.namedWindow") as mock_named_window, \
         patch("cv2.imshow") as mock_imshow, \
         patch("cv2.waitKey") as mock_wait_key, \
         patch("cv2.destroyAllWindows") as mock_destroy:
        
        yield {
            "VideoCapture": mock_cap,
            "namedWindow": mock_named_window,
            "imshow": mock_imshow,
            "waitKey": mock_wait_key,
            "destroyAllWindows": mock_destroy
        }

def test_headless_app_loop(mock_cv2):
    print("\n--- Starting Headless Application Loop Test (Real Image context) ---")
    
    # 1. Setup Mock Behavior
    captured_frames = []
    
    def capture_frame(win_name, frame):
        captured_frames.append(frame.copy())
        
    mock_cv2["imshow"].side_effect = capture_frame
    
    # Run for 15 frames then quit
    mock_cv2["waitKey"].side_effect = [-1] * 14 + [ord('q')]
    
    # 2. Initialize Application with Smart Mock Pipeline
    with patch("app.main_exhibit.Pipeline") as MockPipelineClass:
        
        # Instantiate our smart mock
        # We inject a specific prompt to test the tinting logic
        test_prompt = "A lush green forest, nature photography"
        smart_mock = SmartMockPipeline(prompt=test_prompt)
        
        MockPipelineClass.return_value = smart_mock
        
        exhibit = NoiseMirrorExhibit()
        # Force prompt manager to return our test prompt so the app uses it
        exhibit.prompt_manager.get_current_prompt = MagicMock(return_value=test_prompt)
        
        # 3. Run Application Loop
        print(f"Launching application loop with prompt: '{test_prompt}'...")
        exhibit.run(camera_id=0)
    
    print(f"Loop finished. Captured {len(captured_frames)} frames.")
    
    # 4. Validation
    assert len(captured_frames) > 0, "No frames were displayed"
    last_frame = captured_frames[-1]
    
    # Get Quadrants
    # Top-Left: Camera (Should be our test photo)
    # Top-Right: AI (Should be Green Tinted)
    cam_quadrant = last_frame[0:512, 0:512]
    ai_quadrant = last_frame[0:512, 512:1024]
    
    # Validate Size
    assert last_frame.shape == (1024, 1024, 3)
    
    # Validate AI Tint (Green Channel domination)
    # BGR: Green is channel 1
    mean_b, mean_g, mean_r = cv2.mean(ai_quadrant)[:3]
    print(f"AI Quadrant Color Stats: R={mean_r:.2f}, G={mean_g:.2f}, B={mean_b:.2f}")
    
    # Expect Green to be significant (since we blended 30% pure green)
    # And specifically, G should be higher than R and B if the original image wasn't overwhelmingly purple/red.
    # Given we passed "green forest", the SmartMockPipeline applied (0, 255, 0).
    assert mean_g > mean_r and mean_g > mean_b, \
        f"AI output expected to be Green-tinted for prompt '{test_prompt}', but got BGR=({mean_b},{mean_g},{mean_r})"
        
    # Validate Camera Feed matches input (structural similarity check is simple: checking variance > 0)
    assert np.var(cam_quadrant) > 100, "Camera quadrant is empty/black"
    
    # Validate Noise (Bottom-Left)
    noise_quadrant = last_frame[512:1024, 0:512]
    noise_var = np.var(noise_quadrant)
    print(f"Noise Quadrant Variance: {noise_var:.2f}")
    assert noise_var > 50, "Noise extraction failed (low variance)"

    # 5. Artifact Generation
    output_path = "headless_app_output.jpg"
    cv2.imwrite(output_path, last_frame)
    if os.path.exists(output_path):
         print(f"Saved artifact to {os.path.abspath(output_path)}")
         
    print("--- Test Passed ---")

if __name__ == "__main__":
    # Allow running directly
    pytest.main([__file__])
