# The Noise Mirror

An AI-infused live webcam experience that generates a stylized, dream-like reflection of the user in real-time. This version represents the first iteration (v1) of the Noise Mirror museum exhibit.

## Overview

The Noise Mirror takes a live webcam feed and passes it through an optimized Stable Diffusion pipeline (SDXS) to generate a constantly morphing, stylized version of the video. It presents a 2x2 grid representing the underlying mechanics of diffusion:
- **Top-Left**: Raw webcam feed.
- **Top-Right**: The AI's "dream" interpretation.
- **Bottom-Left**: Spatial noise extracted from the camera feed (motion history).
- **Bottom-Right**: Spatial noise representation matching the AI's generation.

### Key Features
- **Real-Time AI Generation**: Leverages CoreML optimization to achieve >10 FPS on Apple Silicon.
- **Auto-Rotation**: Styles and prompts automatically rotate every 60 seconds (e.g., "watercolor painting", "neon cyberpunk").
- **Manual Override**: Allows operators to manually type prompts by pressing `TAB`.
- **Headless Pipeline Testing**: Implements mock environments for headless CI/CD pipeline validation (`tests/test_headless_app_loop.py`).

## Installation & Setup

### Requirements
- Python 3.10+
- macOS 14+ (Apple Silicon recommended for CoreML Neural Engine acceleration)
- Webcam

### Environment Setup
1. Clone the repository and initialize a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Generate the CoreML Models:
To achieve real-time latency on macOS, the UNet and VAE components must be converted to Apple's `.mlpackage` format.
```bash
python app/convert_for_mirror.py
```
This will download `IDKiro/sdxs-512-0.9` and `taesd`, convert them to CoreML format, and place them in the `models/` directory.

## Usage

### For Developers
Run the core application from the command line:
```bash
python3 -m app.main_exhibit
```
*Note: Due to a recent cleanup, the primary application components reside in `Deployment_v1/app/` if the project was bundled.*

### Packaging for Deployment
The project includes a packaging script to create a portable bundle that can run on any macOS machine without an internet connection or pre-installed dependencies.
```bash
python package_exhibit.py
```
This script will:
- Download a portable Python runtime.
- Bundle `requirements.txt` libraries.
- Pre-download and cache necessary Hugging Face models.
- Create a `Deployment_v1/` directory with a `Launch_Exhibit.command` executable.

## Testing & Validation
The project contains comprehensive tests for verifying both logic and visual quality, located in the `tests/` directory:
- `test_full_pipeline.py`: Verifies identity preservation using CLIP embeddings.
- `test_headless_app_loop.py`: Mocks webcam and UI boundaries to validate the state machine without requiring a monitor.

Run tests using `pytest`:
```bash
pytest tests/
```

## Maintenance & Legacy
Earlier versions of this application have been consolidated into the `Deployment_v1` package during a project structure cleanup. Ensure that new application architectures correctly decouple the UI/Webcam loop (Fast Loop) from the AI Inference loop (Slow Loop) via non-blocking queues, as targeted in the v2 architecture refactor.
