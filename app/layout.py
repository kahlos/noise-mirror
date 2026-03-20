import numpy as np

def build_4_quadrant_grid(cam_raw, ai_raw, cam_noise, ai_noise) -> np.ndarray:
    """
    Combines four 512x512 images into a single 1024x1024 grid.
    Layout:
    [ Cam Raw   | AI Raw    ]
    [ Cam Noise | AI Noise  ]
    """
    top_row = np.hstack((cam_raw, ai_raw))
    bottom_row = np.hstack((cam_noise, ai_noise))
    grid = np.vstack((top_row, bottom_row))
    return grid
