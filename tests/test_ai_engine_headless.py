import pytest
import time
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
from PIL import Image

from app.ai_engine import AIEngine
from app.prompt_manager import PromptManager

@pytest.fixture
def mock_pipeline():
    mock_pipe = MagicMock()
    mock_pipe.to.return_value = mock_pipe
    mock_pipe.scheduler = MagicMock()
    mock_pipe.scheduler.config = {}
    
    dummy_image = Image.new('RGB', (512, 512), color='blue')
    mock_output = MagicMock()
    mock_output.images = [dummy_image]
    
    # Simulated 0.1s inference time
    def side_effect(*args, **kwargs):
        time.sleep(0.1)
        return mock_output
        
    mock_pipe.side_effect = side_effect
    return mock_pipe

@patch('app.ai_engine.LCMScheduler.from_config')
@patch('app.ai_engine.StableDiffusionControlNetPipeline.from_pretrained')
@patch('app.ai_engine.ControlNetModel.from_pretrained')
def test_ai_engine_queue_drop(mock_cn, mock_sd, mock_lcm, mock_pipeline):
    mock_sd.return_value = mock_pipeline
    engine = AIEngine(models_dir="/tmp/models")
    
    frame1 = np.zeros((512, 512, 3), dtype=np.uint8)
    frame2 = np.ones((512, 512, 3), dtype=np.uint8) * 255
    
    engine.process_frame(frame1)
    engine.process_frame(frame2)
    
    assert engine.frame_queue.qsize() == 1
    stored_frame = engine.frame_queue.get()
    assert np.array_equal(stored_frame, frame2), "Stale frame was not dropped"

@patch('app.ai_engine.LCMScheduler.from_config')
@patch('app.ai_engine.StableDiffusionControlNetPipeline.from_pretrained')
@patch('app.ai_engine.ControlNetModel.from_pretrained')
def test_ai_engine_thermal_cap(mock_cn, mock_sd, mock_lcm, mock_pipeline):
    """
    Tests that the slow loop does not exceed 3 FPS.
    """
    mock_sd.return_value = mock_pipeline
    engine = AIEngine(models_dir="/tmp/models")
    pm = MagicMock(spec=PromptManager)
    pm.get_current_prompt.return_value = "headless test prompt"
    
    engine.start(pm)
    
    start_time = time.time()
    results_count = 0
    
    while time.time() - start_time < 1.1:
        engine.process_frame(np.zeros((512, 512, 3), dtype=np.uint8))
        res = engine.get_latest_result(timeout=0.01)
        if res is not None:
            results_count += 1
            
    engine.stop()
    
    # In 1.1 seconds, at 3 FPS cap (0.333s per cycle), it should produce at most 3 results.
    assert results_count <= 3, f"AI Engine exceeded 3 FPS cap, produced {results_count} frames in 1.1s"
    assert results_count >= 2, f"AI Engine is too slow or broken, produced {results_count} frames in 1.1s"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
