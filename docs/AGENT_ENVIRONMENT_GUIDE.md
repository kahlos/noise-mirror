# Agent Environment Guide

This document provides technical instructions for future AI agents or developers to properly run, test, and understand the environment configurations of The Noise Mirror project.

## 1. Python Virtual Environments

The project contains two virtual environment directories, but only one is currently functional and considered the primary environment.

### `.venv` (Primary Environment)
- **Python Version**: 3.12 (via `uv`)
- **State**: Active and functional. Contains streamlined dependencies for Draw Things API integration.
- **Usage**: This is the current recommended environment.
- **Activation**: `source .venv/bin/activate`

## 2. Dependencies

Primary dependencies are listed in `requirements.txt` and focus on the Draw Things API for inference:
- `requests` (for API communication)
- `numpy`, `opencv-python` (for image and video processing)
- `pytest` (for running the test suite)

**Note on Models**: Since switching to the Draw Things API, local model downloads via `diffusers` are no longer required for the main application, though legacy components may still reference the `models/` folder.

## 3. Project Paths & Execution

When executing commands, always ensure your current working directory is the project root (`noise-mirror/`).

### Running the Project
The application should be run as a module from the project root to ensure relative imports resolve correctly:
```bash
# Ensure you are in the project root
source mirror_venv/bin/activate
python3 -m app.main_exhibit
```

### Packaging / Deployment
The script `package_exhibit.py` bundles the app into a portable macOS executable located in the `Deployment_v1/` directory.

## 4. Running the Tests

Tests are located in the `tests/` directory and use `pytest`. The primary environment (`.venv`) already has `pytest` installed. 

### Best Practices for Agents Running Tests
1. **Activate the environment**: 
   ```bash
   source mirror_venv/bin/activate
   ```
2. **Run the entire test suite**:
   ```bash
   pytest tests/
   ```
3. **Run a specific test file**:
   ```bash
   pytest tests/test_video_proxy.py
   ```

### Test Artifacts
Some automated tests (like `test_video_proxy.py` and `test_graphical.py`) generate visual outputs (images and videos). These outputs are configured to be saved inside the `tests/output/` directory. Always check this folder if you need to visually validate the results of a test.

### Headless vs Graphical Tests
- Tests suffixed with `_headless` (e.g., `test_main_exhibit_headless.py`) are designed to run without a display server (great for CI or pure SSH automated agents).
- Some tests might require a display or use dummy video files (e.g., `test_video_proxy.py` uses `test-video.mov` to mock webcam feed). Ensure the proxy video is available if running these.
