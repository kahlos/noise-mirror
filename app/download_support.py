import os
# Set HF_HOME locally to avoid permission errors in global cache
os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".cache", "huggingface")

import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import EulerDiscreteScheduler

def download_support_models(target_dir="models/sdxs_support"):
    """
    Downloads only the Text Encoder, Tokenizer, and Scheduler for SDXS.
    Saves them to a local directory that looks like a diffusers pipeline.
    """
    model_id = "IDKiro/sdxs-512-0.9"
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Downloading support models for {model_id} to {target_dir}...")
    
    # 1. Tokenizer
    print("  Fetching Tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    tokenizer.save_pretrained(target_dir + "/tokenizer")
    
    # 2. Text Encoder
    print("  Fetching Text Encoder...")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    text_encoder.save_pretrained(target_dir + "/text_encoder")
    
    # 3. Scheduler
    print("  Fetching Scheduler...")
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    scheduler.save_pretrained(target_dir + "/scheduler")
    
    # 4. Create a dummy model_index.json to fool from_pretrained
    # This is the critical trick to make StableDiffusionPipeline.from_pretrained work
    # on a partial folder.
    import json
    index_config = {
        "_class_name": "StableDiffusionPipeline",
        "_diffusers_version": "0.19.0",
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "scheduler": ["diffusers", "EulerDiscreteScheduler"],
        "unet": [None, None],
        "vae": [None, None],
        "safety_checker": [None, None],
        "feature_extractor": [None, None]
    }
    
    with open(os.path.join(target_dir, "model_index.json"), "w") as f:
        json.dump(index_config, f, indent=2)
        
    print("Success! Support models downloaded.")

if __name__ == "__main__":
    download_support_models()
