import numpy as np
import cv2


def apply_paint(image_np, mask, target_rgb, strength=0.85):

    """
    Realistic paint engine v2:
    - preserves lighting (L channel)
    - modifies only chroma (A/B channels)
    - smooth mask blending
    """

    if mask is None:
        return image_np

    # --------------------------------------------------
    # 1. Normalize mask → soft alpha
    # --------------------------------------------------
    alpha = mask.astype(np.float32) / 255.0

    # soften edges for realism (paint bleed effect)
    alpha = cv2.GaussianBlur(alpha, (21, 21), 0)

    alpha = np.clip(alpha, 0, 1) * strength

    # --------------------------------------------------
    # 2. Convert to LAB color space
    # --------------------------------------------------
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)

    L, A, B = cv2.split(lab)

    # --------------------------------------------------
    # 3. Target color in LAB
    # --------------------------------------------------
    target = np.uint8([[target_rgb]])
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)[0][0]

    # --------------------------------------------------
    # 4. Blend ONLY chroma channels
    # --------------------------------------------------
    A = A.astype(np.float32)
    B = B.astype(np.float32)

    A = A * (1 - alpha) + target_lab[1] * alpha
    B = B * (1 - alpha) + target_lab[2] * alpha

    # --------------------------------------------------
    # 5. Reconstruct LAB image
    # --------------------------------------------------
    lab_out = cv2.merge([
        L,  # preserve lighting completely
        A.astype(np.uint8),
        B.astype(np.uint8)
    ])

    rgb_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)

    # --------------------------------------------------
    # 6. Final subtle blending for realism
    # --------------------------------------------------
    final = (image_np * (1 - alpha[..., None]) + rgb_out * alpha[..., None])

    return final.astype(np.uint8)
