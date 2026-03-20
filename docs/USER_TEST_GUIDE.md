# User Test Guide for "The Noise Mirror"

This guide outlines the steps to verify the stability, functionality, and performance of "The Noise Mirror" museum exhibit.

## 1. Hardware Setup Verification
- [ ] **Camera Connection**: Ensure a USB webcam is connected.
- [ ] **Lighting**: The exhibit is designed for museum lighting. Ensure the room is not pitch black.
- [ ] **Display**: Connect a 1080p or 4K monitor.

## 2. Launching the Exhibit

### Option A: For Developers (Source Code)
1. Open a terminal in the project root (`noise-mirror/`).
2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
3. Run the application as a module (to fix import paths):
   ```bash
   python3 -m app.main_exhibit
   ```

### Option B: For End Users (Deployment)
1. Navigate to the `Deployment_v1` folder.
2. Double-click `Launch_Exhibit.command` (or run `./Deployment_v1/Launch_Exhibit.command` in terminal).
3. **Note**: The first launch may be slow as MacOS verifies signatures.

### Verification
4. Verify that a window titled "The Noise Mirror - Museum Exhibit" appears.
5. Verify the window resolution is roughly 1024x1024 (plus title bars).

## 3. Visual Verification (The 4 Quadrants)
The display should show a 2x2 grid:
- **Top-Left (Camera Raw)**: A resized 512x512 feed from your webcam.
- **Top-Right (AI Raw)**: The AI-generated dream version of your camera feed.
- **Bottom-Left (Camera Noise)**: A ghostly, high-contrast grey noise map of your camera feed.
- **Bottom-Right (AI Noise)**: The matching noise map from the AI generation.

**Success Criteria:**
- Movement in front of the camera reflects immediately in Top-Left and Bottom-Left.
- The AI quadrants (Right side) follow your movement but with a delay/dream-like quality.
- The Noise maps (Bottom row) should mostly be grey (128) with white/black ghosts where edges move.

## 4. Interaction Testing
### Auto-Rotation
- Wait for 60 seconds (default interval).
- **Verify**: The status text in the top-left corner updates to a new prompt (e.g., "watercolor painting...").
- **Verify**: The visual style of the AI output changes.

### Manual Override
1. Press `TAB`.
2. **Verify**: Processing pauses. An overlay appears: "OVERRIDE ACTIVE".
3. Check the terminal window. It should prompt: `Enter prompt:`.
4. Type `neon cyberpunk city` and press Enter.
5. **Verify**: The status text says `[MANUAL] neon cyberpunk city`.
6. **Verify**: The AI output looks like a neon city.

### Quit
1. Press `Q`.
2. **Verify**: The application closes cleanly without hanging.

## 5. Performance Check
- The application should run smoothly.
- On Apple Silicon (M1/M2/M3), the webcam feed should be fluid (30 FPS), while the AI dream state will update at approximately 3 FPS due to deliberate thermal throttling.

## 6. Troubleshooting
- **Black Screen**: Check camera permissions in System Settings -> Privacy & Security -> Camera.
- **Crash on Start**: Ensure your virtual environment is activated and you have internet access on first launch to download the Hugging Face models into the `models/` directory.
