import os
import time
import gc
import numpy as np
import torch
import coremltools as ct
from diffusers import StableDiffusionPipeline, AutoencoderTiny

# Constraints from Milestone 1
COMPUTE_UNITS = ct.ComputeUnit.CPU_AND_GPU
PRECISION = np.float16

def convert_unet(model_id, save_path):
    print(f"Loading UNet from {model_id}...")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16"
        )
    except Exception:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
    
    unet = pipe.unet.eval().float().cpu()

    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, sample, timestep, encoder_hidden_states):
            return self.unet(
                sample, timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]

    wrapper = UNetWrapper(unet).eval()

    # Input shapes for SDXS-512
    sample = torch.randn(1, 4, 64, 64)
    timestep = torch.tensor([999.0])
    hidden_states = torch.randn(1, 77, 1024)

    print("Tracing UNet...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (sample, timestep, hidden_states))

    print(f"Converting UNet to CoreML (units={COMPUTE_UNITS})...")
    t0 = time.time()
    model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="sample", shape=sample.shape, dtype=PRECISION),
            ct.TensorType(name="timestep", shape=timestep.shape, dtype=PRECISION),
            ct.TensorType(name="encoder_hidden_states", shape=hidden_states.shape, dtype=PRECISION),
        ],
        outputs=[
            ct.TensorType(name="noise_pred", dtype=PRECISION),
        ],
        compute_units=COMPUTE_UNITS,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    print(f"UNet converted in {time.time() - t0:.1f}s")
    model.save(save_path)
    
    del unet, wrapper, traced, model
    gc.collect()

def convert_vae_decoder(save_path):
    print("Loading TinyVAE (TAESD)...")
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").eval().float().cpu()

    class DecoderWrapper(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.decoder = v.decoder

        def forward(self, x):
            return self.decoder(x)

    wrapper = DecoderWrapper(vae).eval()
    # 512 / 8 = 64
    dummy = torch.randn(1, 4, 64, 64)

    print("Tracing VAE Decoder...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy)

    print(f"Converting VAE Decoder to CoreML (units={COMPUTE_UNITS})...")
    t0 = time.time()
    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="latent", shape=dummy.shape, dtype=PRECISION)],
        outputs=[ct.TensorType(name="image", dtype=PRECISION)],
        compute_units=COMPUTE_UNITS,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    print(f"VAE Decoder converted in {time.time() - t0:.1f}s")
    model.save(save_path)

    del vae, wrapper, traced, model
    gc.collect()

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    
    unet_out = "models/unet_sdxs_512.mlpackage"
    vae_out = "models/taesd_decoder.mlpackage"
    
    print("Starting SDXS-512 CoreML Conversion...")
    convert_unet("IDKiro/sdxs-512-0.9", unet_out)
    convert_vae_decoder(vae_out)
    print("Done!")
