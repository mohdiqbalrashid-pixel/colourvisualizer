import numpy as np
import cv2


def create_wall_mask(image_np, seed_point):

    h, w = image_np.shape[:2]

    # --------------------------------------------------
    # Step 1: Edge detection (structure awareness)
    # --------------------------------------------------
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 60, 160)

    # Dilate edges slightly to strengthen boundaries
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    edge_block = edges == 0  # valid growth regions

    # --------------------------------------------------
    # Step 2: Flood fill (but constrained)
    # --------------------------------------------------
    mask = np.zeros((h + 2, w + 2), np.uint8)

    bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    x, y = seed_point

    cv2.floodFill(
        bgr,
        mask,
        (x, y),
        (255, 255, 255),
        loDiff=(25, 25, 25),
        upDiff=(25, 25, 25),
        flags=4
    )

    raw = mask[1:-1, 1:-1].astype(np.uint8)

    # --------------------------------------------------
    # Step 3: Edge-aware constraint (key upgrade)
    # --------------------------------------------------
    constrained = raw * edge_block

    # --------------------------------------------------
    # Step 4: Morphological cleanup (removes noise)
    # --------------------------------------------------
    kernel = np.ones((5, 5), np.uint8)

    cleaned = cv2.morphologyEx(constrained, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # --------------------------------------------------
    # Step 5: Soft mask enhancement (important for painting)
    # --------------------------------------------------
    soft = cv2.GaussianBlur(cleaned.astype(np.float32), (11, 11), 0)

    soft = soft / (soft.max() + 1e-6)

    soft = (soft * 255).astype(np.uint8)

    return soft
