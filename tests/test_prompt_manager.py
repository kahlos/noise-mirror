import pytest
import time
from unittest.mock import patch
from app.prompt_manager import PromptManager

def test_prompt_rotation_logic():
    # Initialize with short interval for testing
    pm = PromptManager(rotation_interval=60.0)
    
    initial_prompt = pm.get_current_prompt()
    print(f"\nInitial Prompt: {initial_prompt}")
    
    # 1. Test no rotation before threshold
    with patch('time.time') as mock_time:
        start_time = 1000.0
        mock_time.return_value = start_time
        pm.last_rotation_time = start_time
        
        # Advance by 30 seconds
        mock_time.return_value = start_time + 30.0
        rotated = pm.update()
        
        assert not rotated
        assert pm.get_current_prompt() == initial_prompt
        print("Verified: No rotation at 30s")

        # 2. Test rotation after threshold
        mock_time.return_value = start_time + 65.0
        rotated = pm.update()
        
        assert rotated
        assert pm.get_current_prompt() != initial_prompt
        assert pm.get_current_prompt() == pm.prompts[1]
        print(f"Verified: Rotated to next prompt at 65s: {pm.get_current_prompt()}")

def test_manual_override():
    pm = PromptManager(rotation_interval=60.0)
    pm.set_manual_override("Museum Masterpiece")
    
    assert pm.get_current_prompt() == "Museum Masterpiece"
    assert pm.is_paused == True
    
    # Update should not rotate even after long time
    with patch('time.time') as mock_time:
        mock_time.return_value = time.time() + 1000.0
        rotated = pm.update()
        assert not rotated
        assert pm.get_current_prompt() == "Museum Masterpiece"
    print("Verified: Manual override prevents auto-rotation")

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
