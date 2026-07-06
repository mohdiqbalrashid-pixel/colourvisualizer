import numpy as np
import cv2


def create_wall_mask(image_np, seed_point):

    """
    Expands a single click into a wall region using flood fill.
    """

    h, w = image_np.shape[:2]

    mask = np.zeros((h + 2, w + 2), np.uint8)

    floodfilled = image_np.copy()

    x, y = seed_point

    # OpenCV floodFill requires BGR
    bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    cv2.floodFill(
        bgr,
        mask,
        (x, y),
        (255, 0, 0),
        loDiff=(20, 20, 20),
        upDiff=(20, 20, 20),
        flags=4
    )

    result = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Extract filled region as mask
    final_mask = mask[1:-1, 1:-1]

    final_mask = (final_mask * 255).astype(np.uint8)

    return final_mask, result
