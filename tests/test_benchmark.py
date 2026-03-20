import pytest
import time
import numpy as np
import cv2
from app.cv_engine import extract_noise

def test_cv_engine_performance():
    """
    Benchmarks the CV engine to ensure it runs well within the 30fps budget (33ms).
    """
    # 1. Setup
    frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    iterations = 100
    
    # 2. Warmup
    for _ in range(10):
        extract_noise(frame)
        
    # 3. Benchmark
    start_time = time.time()
    for _ in range(iterations):
        extract_noise(frame)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / iterations
    fps = 1.0 / avg_time
    
    print(f"\nBenchmark Results:")
    print(f"Total Time ({iterations} iters): {total_time:.4f}s")
    print(f"Average Time per Frame: {avg_time*1000:.2f}ms")
    print(f"Estimated CV Engine FPS: {fps:.2f}")
    
    # 4. Assertions
    # Must be faster than 10ms to leave room for AI Inference
    # The AI takes ~600ms on Neural Engine, but CV must be negligible.
    assert avg_time < 0.010, f"CV Engine too slow: {avg_time*1000:.2f}ms (Target < 10ms)"

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
