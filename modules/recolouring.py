from __future__ import annotations

import cv2
import numpy as np

from modules.config import DEFAULT_PAINT_STRENGTH


def apply_paint(
    image_np: np.ndarray,
    mask: np.ndarray,
    target_rgb: tuple[int, int, int],
    strength: float = DEFAULT_PAINT_STRENGTH,
) -> np.ndarray:
    """
    Lighting-aware paint engine.

    This version:
    - preserves luminance and wall texture
    - strengthens weak mask edges
    - avoids the double-blending issue that caused pale boundaries
    - keeps shadows and highlights from the original image
    """

    if image_np is None or mask is None:
        return image_np

    alpha = _create_edge_recovery_alpha(mask, strength)

    if not np.any(alpha > 0):
        return image_np

    image_lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)

    target_rgb_array = np.uint8([[target_rgb]])
    target_lab = cv2.cvtColor(target_rgb_array, cv2.COLOR_RGB2LAB)[0][0]

    lightness, channel_a, channel_b = cv2.split(image_lab)

    channel_a = channel_a.astype(np.float32)
    channel_b = channel_b.astype(np.float32)

    channel_a = channel_a * (1 - alpha) + float(target_lab[1]) * alpha
    channel_b = channel_b * (1 - alpha) + float(target_lab[2]) * alpha

    painted_lab = cv2.merge(
        [
            lightness,
            np.clip(channel_a, 0, 255).astype(np.uint8),
            np.clip(channel_b, 0, 255).astype(np.uint8),
        ]
    )

    painted_rgb = cv2.cvtColor(painted_lab, cv2.COLOR_LAB2RGB)

    return painted_rgb.astype(np.uint8)


def _create_edge_recovery_alpha(mask: np.ndarray, strength: float) -> np.ndarray:
    base = mask.astype(np.float32) / 255.0
    base = np.clip(base, 0, 1)

    binary = (base > 0.05).astype(np.uint8)

    if not np.any(binary > 0):
        return np.zeros_like(base, dtype=np.float32)

    core_kernel = np.ones((3, 3), dtype=np.uint8)
    expand_kernel = np.ones((5, 5), dtype=np.uint8)

    core = cv2.erode(binary, core_kernel, iterations=1).astype(np.float32)
    expanded = cv2.dilate(binary, expand_kernel, iterations=1).astype(np.float32)

    edge_recovery_ring = np.clip(expanded - core, 0, 1)

    soft = cv2.GaussianBlur(base, (21, 21), 0)
    soft = np.clip(soft, 0, 1)

    alpha = (
        0.58 * soft
        + 0.32 * binary.astype(np.float32)
        + 0.10 * edge_recovery_ring
    )

    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
    alpha = np.clip(alpha, 0, 1)

    return alpha * float(strength)
