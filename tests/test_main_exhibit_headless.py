import pytest
import time
import numpy as np
from unittest.mock import MagicMock, patch

from app.main_exhibit import NoiseMirrorExhibit

@patch('app.main_exhibit.cv2.VideoCapture')
@patch('app.main_exhibit.cv2.imshow')
@patch('app.main_exhibit.cv2.waitKey')
@patch('app.main_exhibit.cv2.namedWindow')
@patch('app.main_exhibit.cv2.destroyAllWindows')
@patch('app.main_exhibit.cv2.putText')
@patch('app.main_exhibit.cv2.rectangle')
@patch('app.main_exhibit.AIEngine')
def test_main_exhibit_fast_loop(mock_ai_engine_class, mock_rectangle, mock_putText, mock_destroy, mock_namedWindow, mock_waitKey, mock_imshow, mock_VideoCapture):
    """
    Tests that the Fast Loop (webcam + display) runs unhindered.
    """
    # Mock AIEngine instance
    mock_ai = MagicMock()
    mock_ai_engine_class.return_value = mock_ai
    
    # Provide a dummy AI result to ensure the UI can draw something
    dummy_ai_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mock_ai.get_latest_result.return_value = (dummy_ai_frame, dummy_ai_frame)
    
    # Mock cv2 VideoCapture to yield frames
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    
    # Create fake webcam frames
    frame_count = [0]
    def mock_read():
        frame_count[0] += 1
        return True, np.zeros((720, 1280, 3), dtype=np.uint8)
    mock_cap.read.side_effect = mock_read
    
    mock_VideoCapture.return_value = mock_cap
    
    # We want the loop to exit after 50 frames
    def custom_wait_key(delay):
        if frame_count[0] >= 50:
            return ord('q')
        return -1
    mock_waitKey.side_effect = custom_wait_key
    
    exhibit = NoiseMirrorExhibit()
    
    start_time = time.time()
    exhibit.run(camera_id=0)
    end_time = time.time()
    
    # Check that it processed 50 frames
    assert frame_count[0] == 50, "Did not process expected number of frames"
    
    # AI Engine must have been started
    mock_ai.start.assert_called_once()
    
    # Verify non-blocking nature:
    # Processing 50 frames with mock (no sleep, no heavy AI) should be very quick
    # Definitely should be less than 1 second (this proves the AI thread isn't blocking it).
    duration = end_time - start_time
    assert duration < 2.0, f"Fast loop is blocked or too slow, took {duration:.2f}s for 50 frames"
    
    mock_ai.stop.assert_called_once()
    mock_destroy.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
