import time

class PromptManager:
    def __init__(self, rotation_interval=60.0):
        self.prompts = [
            "oil painting style, masterpiece, highly detailed",
            "watercolor painting, soft brushstrokes, artistic",
            "pencil sketch, detailed line art, cross-hatching",
            "cyberpunk neon city, futuristic, glowing lights",
            "studio ghibli anime style, beautiful landscape"
        ]
        self.current_index = 0
        self.rotation_interval = rotation_interval
        self.last_rotation_time = time.time()
        self.is_paused = False
        self.manual_prompt = None

    def get_current_prompt(self):
        if self.manual_prompt:
            return self.manual_prompt
        return self.prompts[self.current_index]

    def update(self):
        """Checks if it's time to rotate the prompt. Returns True if rotated."""
        if self.is_paused or self.manual_prompt:
            return False
            
        if time.time() - self.last_rotation_time > self.rotation_interval:
            self.current_index = (self.current_index + 1) % len(self.prompts)
            self.last_rotation_time = time.time()
            return True
        return False

    def set_manual_override(self, prompt):
        self.manual_prompt = prompt
        self.is_paused = True # Pause auto-rotation when manual is set
        self.last_rotation_time = time.time() # Reset timer for when/if we resume

    def resume_auto_rotation(self):
        self.manual_prompt = None
        self.is_paused = False
        self.last_rotation_time = time.time()

    def pause_rotation(self):
        self.is_paused = True

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if not self.is_paused:
            self.last_rotation_time = time.time()
