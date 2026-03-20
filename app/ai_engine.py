import os
import cv2
import time
import queue
import threading
import torch
import gc
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
from app.cv_engine import extract_noise

class AIEngine:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        
        # Load ControlNet
        print("Loading ControlNet...")
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            cache_dir=self.models_dir,
            torch_dtype=torch.float16
        )
        
        # Load base SD 1.5 pipeline but attach the ControlNet
        print("Loading Stable Diffusion Pipeline...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            controlnet=self.controlnet,
            cache_dir=self.models_dir,
            safety_checker=None,
            torch_dtype=torch.float16
        ).to("mps") # Native Apple Silicon GPU target
        
        # Silence MPSGraph cache permission spam by setting a local cache dir
        os.makedirs(os.path.join(self.models_dir, ".mtl_cache"), exist_ok=True)
        os.environ["MTL_SHADER_CACHE_DIR"] = os.path.join(self.models_dir, ".mtl_cache")
        
        # Inject LCM (Latent Consistency Model) for fast 4-step generation
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", weight_name="pytorch_lora_weights.safetensors", cache_dir=self.models_dir)

        
        # Thread communication
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.running = False
        self.prompt_manager = None
        self.thread = None

    def start(self, prompt_manager):
        self.running = True
        self.prompt_manager = prompt_manager
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

    def process_frame(self, frame):
        """Put frame into the Slow Loop queue (non-blocking, dropping stale)."""
        if not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(frame)

    def get_latest_result(self, timeout=None):
        """Retrieve the latest completely rendered AI frame & noise."""
        if timeout is None:
            if self.result_queue.empty():
                return None
            try:
                return self.result_queue.get_nowait()
            except queue.Empty:
                return None
        else:
            try:
                return self.result_queue.get(timeout=timeout)
            except queue.Empty:
                return None

    def _inference_loop(self):
        print("AI Inference Slow Loop started.")
        while self.running:
            start_time = time.time()
            
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if not self.prompt_manager:
                continue

            current_prompt = self.prompt_manager.get_current_prompt()
            
            # Pre-process (Canny Edge Detection for ControlNet)
            frame_resized = cv2.resize(frame, (512, 512))
            webcam_frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            canny_edges = cv2.Canny(webcam_frame_gray, 100, 200)
            
            # ControlNet requires a 3-channel RGB image
            canny_edges_rgb = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2RGB)
            canny_edges_pil = Image.fromarray(canny_edges_rgb)
            
            # Inference (LCM + ControlNet)
            try:
                ai_image_pil = self.pipe(
                    prompt=current_prompt,
                    image=canny_edges_pil,          # ControlNet acts as the structural foundation
                    num_inference_steps=4,          # LCM allows for 4-step blazing fast generation
                    guidance_scale=1.5,             # LCM requires very low guidance scales
                    controlnet_conditioning_scale=0.8
                ).images[0]
                
                # Convert PIL to CV2 BGR
                ai_raw = cv2.cvtColor(np.array(ai_image_pil), cv2.COLOR_RGB2BGR)
                
                # Post-process (extract noise from the AI generated frame)
                ai_noise = extract_noise(ai_raw)
                
                # Publish to State Fetch Queue
                if not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.result_queue.put((ai_raw, ai_noise))
                
            except Exception as e:
                print(f"Inference error: {e}")
            finally:
                if torch.backends.mps.is_available():
                    gc.collect()
                    torch.mps.empty_cache()
            
            # Thermal Management
            # Capping AI at strict 3 FPS (0.333s per cycle)
            elapsed = time.time() - start_time
            target_fps_time = 0.333
            if elapsed < target_fps_time:
                time.sleep(target_fps_time - elapsed)
