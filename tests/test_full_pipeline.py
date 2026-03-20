
import os
import sys
import tempfile

# Set environment variables for cache stability
local_tmp = os.path.join(os.getcwd(), ".tmp")
os.makedirs(local_tmp, exist_ok=True)
os.environ["TMPDIR"] = local_tmp
os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".cache", "huggingface")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

import torch
import numpy as np
import cv2
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
from diffusers import AutoPipelineForImage2Image
from transformers import CLIPProcessor, CLIPModel

# Ensure local app is in path
sys.path.append(os.getcwd())

# --- 1. Infrastructure Setup ---

class VerificationModel:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        print("\n[VerificationModel] Loading CLIP model...")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).to(torch.float16)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def calculate_similarity(self, image_bgr, prompt):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        inputs = self.processor(text=[prompt], images=pil_image, return_tensors="pt", padding=True).to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            score = logits_per_image.item()
        return score

# --- 2. Test Class ---

class TestFullImg2ImgPipeline:
    
    @pytest.fixture(scope="class")
    def verifier(self):
        return VerificationModel.get_instance()

    @pytest.fixture(scope="class")
    def diffusion_pipe(self):
        print("\n[TestFullImg2ImgPipeline] Loading SDXS for Img2Img Test (PyTorch/MPS)...")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model_id = "IDKiro/sdxs-512-0.9" 
        
        try:
            pipe = AutoPipelineForImage2Image.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                variant="fp16"
            ).to(device)
            pipe.set_progress_bar_config(disable=True)
            return pipe
        except Exception as e:
            print(f"Failed to load SDXS: {e}. Falling back to SD-Turbo.")
            pipe = AutoPipelineForImage2Image.from_pretrained(
                "stabilityai/sd-turbo", 
                torch_dtype=torch.float16, 
                variant="fp16"
            ).to(device)
            pipe.set_progress_bar_config(disable=True)
            return pipe

    def test_img2img_prompt_consistency(self, verifier, diffusion_pipe):
        """
        Runs a real Image-to-Image transition and verifies consistency with CLIP.
        """
        # Load real test image
        img_path = os.path.join(os.getcwd(), "test-photo.jpg")
        if not os.path.exists(img_path):
             # Create a dummy image if not exists
             input_img = Image.new('RGB', (512, 512), color='gray')
             print("WARNING: test-photo.jpg not found, using dummy gray image.")
        else:
             input_img = Image.open(img_path).convert("RGB").resize((512, 512))
        
        test_cases = [
            ("A photo of the original person but stylized as a Vincent van Gogh painting, thick brushstrokes, maintains perfect facial identity", "Van Gogh Identification"),
            ("A high-quality pencil sketch of the original person, hand-drawn lines, charcoal on paper, recognizable face", "Sketch Identification"),
            ("Subtle beautification of the original person, glowing skin, portrait lighting, high likeness", "Beautification Identification")
        ]
        
        for prompt, short_name in test_cases:
            print(f"\n--- Identity-Preserving Segment: {prompt} ---")
            
            # Run Inference
            with torch.no_grad():
                # 0.42 is the refined 'mirror' strength
                output_pil = diffusion_pipe(
                    prompt=prompt, 
                    image=input_img,
                    strength=0.42,
                    num_inference_steps=12,
                    guidance_scale=0.0
                ).images[0]
            
            output_bgr = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGB2BGR)
            
            # Save result BEFORE assertion
            out_fn = f"identity_test_{short_name.lower().replace(' ', '_')}.jpg"
            cv2.imwrite(out_fn, output_bgr)
            print(f"Saved identity-preserving result to {out_fn}")

            # 1. Verify Layout Metric
            assert output_bgr.shape == (512, 512, 3)
            
            # 2. Verify Semantic Consistency
            # Use distractors that are completely unrelated to art or people
            distractors = ["a macro photo of a circuit board", "a heap of gravel and dust", "a close up of white rice"]
            all_prompts = [prompt] + distractors
            
            scores = []
            for p in all_prompts:
                s = verifier.calculate_similarity(output_bgr, p)
                scores.append((p, s))
                print(f"Score for '{p[:30]}...': {s:.2f}")
            
            scores.sort(key=lambda x: x[1], reverse=True)
            top_prompt, top_score = scores[0]
            
            assert top_prompt == prompt, f"Image for '{prompt}' was not identified correctly by CLIP. Identified as '{top_prompt}'"
            print(f"Result: SUCCESS - Correctly identified as {short_name}")

            # Save result
            out_fn = f"img2img_science_{short_name.lower().replace('/', '_')}.jpg"
            cv2.imwrite(out_fn, output_bgr)

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
