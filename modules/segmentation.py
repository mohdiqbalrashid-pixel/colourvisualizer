import numpy as np
import cv2


def create_wall_mask(image_np, seed_point):

    h, w = image_np.shape[:2]

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

    final_mask = mask[1:-1, 1:-1] * 255

    final_mask = final_mask.astype(np.uint8)

    return final_mask
