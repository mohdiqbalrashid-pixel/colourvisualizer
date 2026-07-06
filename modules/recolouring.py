import numpy as np
import cv2


def apply_paint(image_np, mask, target_rgb, strength=0.85):

    """
    Realistic paint simulation:
    - preserves lighting
    - preserves texture
    - blends in LAB space
    """

    if mask is None:
        return image_np

    # Normalize mask to 0–1
    soft_mask = mask.astype(np.float32) / 255.0

    soft_mask = cv2.GaussianBlur(soft_mask, (21, 21), 0)

    soft_mask = np.clip(soft_mask, 0, 1)

    # Convert image to LAB
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)

    # Target colour in LAB
    target = np.uint8([[target_rgb]])
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)[0][0]

    L, A, B = cv2.split(lab)

    # Blend chroma channels
    A = A.astype(np.float32)
    B = B.astype(np.float32)

    A = A * (1 - soft_mask * strength) + target_lab[1] * (soft_mask * strength)
    B = B * (1 - soft_mask * strength) + target_lab[2] * (soft_mask * strength)

    lab_final = cv2.merge([
        L,
        A.astype(np.uint8),
        B.astype(np.uint8)
    ])

    return cv2.cvtColor(lab_final, cv2.COLOR_LAB2RGB)
