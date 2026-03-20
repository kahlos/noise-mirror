# The Noise Mirror

An AI-infused live webcam experience that generates a stylized, dream-like reflection of the user in real-time. This version represents the V2 architecture of the Noise Mirror museum exhibit.

## Overview

The Noise Mirror takes a live webcam feed and passes it through an optimized Stable Diffusion pipeline to generate a constantly morphing, stylized version of the video. It presents a 2x2 grid representing the underlying mechanics of diffusion:
- **Top-Left**: Raw webcam feed.
- **Top-Right**: The AI's "dream" interpretation.
- **Bottom-Left**: Spatial noise extracted from the camera feed (motion history).
- **Bottom-Right**: Spatial noise representation matching the AI's generation.

### Key Features
- **Async AI Generation**: Decouples the UI/Webcam loop (Fast Loop @ 30 FPS) from the AI Inference loop (Slow Loop @ 3 FPS target) via non-blocking queues.
- **Optimized Pipeline**: Uses PyTorch `diffusers` with `StableDiffusionControlNetPipeline`, `LCMScheduler`, and native Apple Silicon GPU (`mps`) backend for blazing fast generation.
- **Auto-Rotation**: Styles and prompts automatically rotate every 60 seconds (e.g., "watercolor painting", "neon cyberpunk").
- **Manual Override**: Allows operators to manually type prompts by pressing `TAB`.
- **Thermal Management**: Caps AI inference at a strict 3 FPS to maintain stability during continuous museum exhibition.

## Installation & Setup

### Requirements
- Python 3.10+
- macOS 14+ (Apple Silicon highly recommended for `mps` Neural Engine acceleration)
- Webcam

### Environment Setup
1. Clone the repository and initialize a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. The application will automatically download the necessary Hugging Face models (`runwayml/stable-diffusion-v1-5`, `lllyasviel/sd-controlnet-canny`, `latent-consistency/lcm-lora-sdv1-5`) to the `models/` directory on first run.

## Usage

### For Developers
Run the core application from the command line:
```bash
python3 -m app.main_exhibit
```
*Note: Legacy v1 structural components have been moved out during the v2 structural cleanup.*

### Packaging for Deployment
The project includes a packaging script to create a portable bundle that can run on any macOS machine without an internet connection or pre-installed dependencies.
```bash
python package_exhibit.py
```
This script will:
- Download a portable Python runtime.
- Bundle `requirements.txt` libraries.
- Pre-download and cache necessary Hugging Face models.
- Create a `Deployment_v1/` directory with a `Launch_Exhibit.command` executable (Note: Directory name kept for backwards compatibility with legacy museum launchers).

## Testing & Validation
The project contains comprehensive tests for verifying both logic and headless visual quality, located in the `tests/` directory:
- `test_video_proxy.py`: Validates visual continuity using a pre-recorded `.mov` feed instead of a live webcam.
- `test_main_exhibit_headless.py`: Mocks webcam and UI boundaries to validate the state machine without requiring a monitor.

Run tests using `pytest`:
```bash
pytest tests/
```
