import cv2
import numpy as np

def extract_noise(frame: np.ndarray) -> np.ndarray:
    """
    Extracts high-frequency noise from a frame using Signed Subtraction and Dynamic Gain Control.
    This prevents visual clipping under varying museum lighting conditions.
    """
    # 1. Cast to float32
    f32 = frame.astype(np.float32)
    
    # 2. Blur frame (5x5 Gaussian)
    blur = cv2.GaussianBlur(f32, (5, 5), 0)
    
    # 3. Calculate raw noise via Signed Subtraction (DO NOT use cv2.absdiff)
    raw_noise = f32 - blur
    
    # 4. Calculate Dynamic Gain Control (AGC)
    # Target standard deviation of 30.0, max gain capped at 50.0
    std = np.std(raw_noise)
    gain = min(50.0, 30.0 / (std + 1e-5))
    
    # 5. Amplify and offset to neutral gray (128.0)
    amplified = (raw_noise * gain) + 128.0
    
    # 6. Clip to 0-255 and cast to uint8
    output = np.clip(amplified, 0, 255).astype(np.uint8)
    
    return output
