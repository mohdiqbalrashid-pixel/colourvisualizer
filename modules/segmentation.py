import numpy as np
import cv2


def create_wall_mask(image_np, seed_point):

    h, w = image_np.shape[:2]

    # --------------------------------------------------
    # 1. Strong adaptive edge detection
    # --------------------------------------------------
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 40, 120)

    # Strengthen edges (critical fix)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    edges = cv2.erode(edges, kernel, iterations=1)

    # Invert edges → allowed region map
    free_space = edges == 0

    # --------------------------------------------------
    # 2. Flood fill base mask
    # --------------------------------------------------
    mask = np.zeros((h + 2, w + 2), np.uint8)

    bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    x, y = seed_point

    cv2.floodFill(
        bgr,
        mask,
        (x, y),
        255,
        loDiff=(15, 15, 15),
        upDiff=(15, 15, 15),
        flags=4
    )

    raw = mask[1:-1, 1:-1].astype(np.uint8)

    # --------------------------------------------------
    # 3. HARD EDGE CONSTRAINT (real fix)
    # --------------------------------------------------
    constrained = raw & free_space.astype(np.uint8)

    # --------------------------------------------------
    # 4. Safety clamp (prevents full-image takeover)
    # --------------------------------------------------
    coverage = np.sum(constrained) / (h * w)

    if coverage > 0.60:
        # fallback: shrink aggressively
        kernel = np.ones((7, 7), np.uint8)
        constrained = cv2.erode(constrained, kernel, iterations=3)

    if coverage < 0.01:
        # fallback: relax slightly if too strict
        constrained = raw

    # --------------------------------------------------
    # 5. Morph cleanup
    # --------------------------------------------------
    kernel = np.ones((5, 5), np.uint8)

    cleaned = cv2.morphologyEx(constrained, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # --------------------------------------------------
    # 6. Soft mask for paint blending
    # --------------------------------------------------
    soft = cv2.GaussianBlur(cleaned.astype(np.float32), (15, 15), 0)

    soft = soft / (soft.max() + 1e-6)

    return (soft * 255).astype(np.uint8)
