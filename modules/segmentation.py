import numpy as np
import cv2


def create_wall_mask(image_np, seed_point):

    h, w = image_np.shape[:2]

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Detect edges (wall boundaries often show here)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Invert edges so flood fill can treat edges as barriers
    edge_mask = (edges == 0).astype(np.uint8)

    # Create flood fill mask (must be 2 pixels larger)
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Copy image for flood fill
    bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    x, y = seed_point

    # We slightly weaken flood fill sensitivity
    lo_diff = (20, 20, 20)
    up_diff = (20, 20, 20)

    # Apply flood fill
    cv2.floodFill(
        bgr,
        mask,
        (x, y),
        (255, 255, 255),
        loDiff=lo_diff,
        upDiff=up_diff,
        flags=4 | (255 << 8)
    )

    raw_mask = mask[1:-1, 1:-1]

    # 🔥 KEY UPGRADE: stop spillover using edges
    raw_mask = raw_mask * edge_mask

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

    cleaned = (cleaned * 255).astype(np.uint8)

    return cleaned
