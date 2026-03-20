import os
import sys
import numpy as np
import cv2
import coremltools as ct
from PIL import Image

# Set local TMPDIR for CoreML compilation in sandbox
os.environ["TMPDIR"] = os.path.join(os.getcwd(), "tmp")
if not os.path.exists(os.environ["TMPDIR"]):
    os.makedirs(os.environ["TMPDIR"])

def test_vae_roundtrip():
    model_path = "Deployment_v1/models"
    enc_path = os.path.join(model_path, "taesd_encoder_512.mlpackage")
    dec_path = os.path.join(model_path, "taesd_decoder.mlpackage")
    
    print(f"Loading Encoder: {enc_path}")
    vae_encoder = ct.models.MLModel(enc_path, compute_units=ct.ComputeUnit.ALL)
    print(f"Loading Decoder: {dec_path}")
    vae_decoder = ct.models.MLModel(dec_path, compute_units=ct.ComputeUnit.ALL)
    
    # Create a dummy image (gradient)
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(512):
        img[i, :, :] = [i // 2, 0, 255 - (i // 2)] # BGR gradient
    
    cv2.imwrite("debug_input.jpg", img)
    
    # Preprocess
    img_in = img.astype(np.float32) / 127.5 - 1.0
    img_in = img_in[:, :, ::-1].transpose(2, 0, 1) # BGR -> RGB -> CHW
    img_in = np.expand_dims(img_in, 0).astype(np.float16) # NCHW
    
    print("Encoding...")
    enc_out = vae_encoder.predict({"image": img_in})
    latents = np.array(enc_out["latent"]).astype(np.float16)
    
    print(f"Latent stats: min={latents.min()}, max={latents.max()}, mean={latents.mean()}")
    
    # Test WITH scaling
    latents_scaled = latents * 0.18215
    
    # Decode WITH unscaling
    print("Decoding (Scaled 0.18215)...")
    dec_in = latents_scaled / 0.18215
    dec_out = vae_decoder.predict({"latent": dec_in})
    
    img_out = np.array(dec_out["image"]).astype(np.float32).squeeze(0).transpose(1, 2, 0) # CHW -> HWC
    img_out = ((img_out + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
    cv2.imwrite("debug_roundtrip_scaled.jpg", img_out)
    
    # Decode WITHOUT unscaling (if we assume model expects raw)
    print("Decoding (Raw)...")
    dec_in_raw = latents
    dec_out_raw = vae_decoder.predict({"latent": dec_in_raw})
    
    img_out_raw = np.array(dec_out_raw["image"]).astype(np.float32).squeeze(0).transpose(1, 2, 0)
    img_out_raw = ((img_out_raw + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    img_out_raw = cv2.cvtColor(img_out_raw, cv2.COLOR_RGB2BGR)
    cv2.imwrite("debug_roundtrip_raw.jpg", img_out_raw)
    
    print("Done. Check debug_roundtrip_scaled.jpg and debug_roundtrip_raw.jpg")

if __name__ == "__main__":
    test_vae_roundtrip()
