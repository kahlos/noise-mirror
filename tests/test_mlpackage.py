import os
import coremltools as ct
import numpy as np

def test_unet_shape():
    unet_path = "models/unet_sdxs_512.mlpackage"
    assert os.path.exists(unet_path), f"UNet not found at {unet_path}"
    
    print(f"\nLoading UNet: {unet_path}")
    model = ct.models.MLModel(unet_path)
    
    # Expected: sample(1,4,64,64), timestep(1), encoder_hidden_states(1,77,1024)
    spec = model.get_spec()
    print("UNet Input Descriptions:")
    for input in spec.description.input:
         name = input.name
         shape = [d for d in input.type.multiArrayType.shape]
         print(f"  {name}: {shape}")
         
         if name == "sample":
             assert shape == [1, 4, 64, 64], f"Unexpected sample shape: {shape}"
         elif name == "timestep":
             assert shape == [1], f"Unexpected timestep shape: {shape}"
         elif name == "encoder_hidden_states":
             assert shape == [1, 77, 1024], f"Unexpected hidden_states shape: {shape}"

    print("UNet Output Descriptions:")
    for output in spec.description.output:
        name = output.name
        print(f"  {output.name}")
        if name == "noise_pred":
             # Shape should be same as sample
             pass

def test_vae_decoder_shape():
    vae_path = "models/taesd_decoder.mlpackage"
    assert os.path.exists(vae_path), f"VAE Decoder not found at {vae_path}"
    
    print(f"\nLoading VAE Decoder: {vae_path}")
    model = ct.models.MLModel(vae_path)
    
    spec = model.get_spec()
    print("VAE Decoder Input Descriptions:")
    for input in spec.description.input:
        name = input.name
        shape = [d for d in input.type.multiArrayType.shape]
        print(f"  {name}: {shape}")
        if name == "latent":
             assert shape == [1, 4, 64, 64], f"Unexpected latent shape: {shape}"

    print("VAE Decoder Output Descriptions:")
    for output in spec.description.output:
        print(f"  {output.name}")

if __name__ == "__main__":
    # If run directly, just execute the logic
    try:
        test_unet_shape()
        test_vae_decoder_shape()
        print("\nVerification Passed!")
    except Exception as e:
        print(f"\nVerification Failed: {e}")
        exit(1)
