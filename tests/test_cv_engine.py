import pytest
import numpy as np
import cv2
from app.cv_engine import extract_noise
from app.layout import build_4_quadrant_grid

def test_extract_noise_math():
    # 1. Create a flat gray (128) array
    base = np.full((512, 512, 3), 128, dtype=np.uint8)
    
    # 2. Inject random normal noise (std = 2.0)
    noise = np.random.normal(0, 2.0, (512, 512, 3))
    input_frame = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # 3. Run extract_noise
    output = extract_noise(input_frame)
    
    # 4. Assertions
    # Input variance calculation (approx 2.0^2 = 4.0)
    in_var = np.var(input_frame)
    out_var = np.var(output)
    
    print(f"\nInput Var: {in_var:.4f}")
    print(f"Output Var: {out_var:.4f}")
    print(f"Mean: {np.mean(output):.4f}")
    
    # Assert output variance increased by at least 10x (due to AGC amplifying to target std 30)
    # Target std 30 -> Var 900. Input var ~4. 900 / 4 = 225x. 10x is safe hurdle.
    assert out_var > in_var * 10, f"Variance increase insufficient: {out_var} vs {in_var}"
    
    # Assert mean is centered around neutral gray
    assert 120 < np.mean(output) < 136, f"Mean offset too high: {np.mean(output)}"

def test_build_4_quadrant_grid_shape():
    # Create four 512x512x3 dummy images
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    grid = build_4_quadrant_grid(img, img, img, img)
    
    assert grid.shape == (1024, 1024, 3), f"Unexpected grid shape: {grid.shape}"

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
