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
    Exact-colour-preserving paint engine.

    Goal:
    - match the selected RGB much more closely
    - preserve sunlight, shadows and wall texture
    - avoid the washed-out / shifted colour effect

    Approach:
    - Use the selected RGB as the base wall colour
    - Build a shading map from the original wall luminance
    - Reapply that luminance variation onto the target colour
    - Blend softly only at the mask boundary
    """

    if image_np is None or mask is None:
        return image_np

    image = _ensure_rgb_uint8(image_np).astype(np.float32) / 255.0
    mask_uint8 = _ensure_mask_uint8(mask)

    alpha = _create_boundary_alpha(mask_uint8, strength)

    if not np.any(alpha > 0):
        return image_np

    target = np.array(target_rgb, dtype=np.float32) / 255.0

    luminance = _relative_luminance(image)
    shading = _build_shading_map(luminance, mask_uint8)

    # Apply original shading to the target colour.
    recoloured = target[None, None, :] * shading[:, :, None]
    recoloured = np.clip(recoloured, 0.0, 1.0)

    # Blend only using the prepared alpha map.
    final = image * (1.0 - alpha[:, :, None]) + recoloured * alpha[:, :, None]
    final = np.clip(final, 0.0, 1.0)

    return (final * 255.0).astype(np.uint8)


def _build_shading_map(luminance: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Build a brightness map from the original wall.

    We want:
    - target RGB in normal wall areas
    - darker target RGB in shadowed areas
    - lighter target RGB in sunlit areas
    """

    mask_bool = mask > 0

    if not np.any(mask_bool):
        return np.ones_like(luminance, dtype=np.float32)

    masked_values = luminance[mask_bool]

    # Use an upper-mid percentile so normal lit wall areas stay close to the exact target RGB,
    # while darker areas remain darker and highlights remain lighter.
    reference_luminance = float(np.percentile(masked_values, 70))

    reference_luminance = max(reference_luminance, 1e-4)

    # Large-scale shading
    broad_ratio = luminance / reference_luminance
    broad_ratio = np.clip(broad_ratio, 0.35, 1.85)

    # Fine texture / local wall variation
    local_base = cv2.GaussianBlur(luminance, (0, 0), sigmaX=7, sigmaY=7)
    local_base = np.clip(local_base, 1e-4, None)

    detail_ratio = luminance / local_base
    detail_ratio = np.clip(detail_ratio, 0.75, 1.25)

    # Keep most of the large-scale lighting, but only a gentle amount of micro detail
    shading = broad_ratio * np.power(detail_ratio, 0.30)
    shading = np.clip(shading, 0.25, 1.95)

    return shading.astype(np.float32)


def _create_boundary_alpha(mask: np.ndarray, strength: float) -> np.ndarray:
    """
    Create strong interior coverage and soft edge blending.

    Interior pixels should repaint strongly so the result matches the chosen RGB.
    Boundaries remain soft to avoid harsh cut lines.
    """

    base = mask.astype(np.float32) / 255.0
    base = np.clip(base, 0.0, 1.0)

    binary = (base > 0.05).astype(np.uint8)

    if not np.any(binary > 0):
        return np.zeros_like(base, dtype=np.float32)

    core_kernel = np.ones((3, 3), dtype=np.uint8)
    expand_kernel = np.ones((5, 5), dtype=np.uint8)

    core = cv2.erode(binary, core_kernel, iterations=1).astype(np.float32)
    expanded = cv2.dilate(binary, expand_kernel, iterations=1).astype(np.float32)

    edge_ring = np.clip(expanded - core, 0.0, 1.0)

    soft = cv2.GaussianBlur(base, (21, 21), 0)
    soft = np.clip(soft, 0.0, 1.0)

    # Stronger interior, softer edges
    alpha = (
        0.50 * soft
        + 0.40 * binary.astype(np.float32)
        + 0.10 * edge_ring
    )

    # Ensure the interior remains strong enough for accurate colour matching
    alpha = np.maximum(alpha, core * 0.98)

    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    alpha = np.clip(alpha, 0.0, 1.0)

    return alpha * float(strength)


def _relative_luminance(image: np.ndarray) -> np.ndarray:
    """
    Perceived luminance from RGB image in 0..1 range.
    """
    return (
        0.2126 * image[:, :, 0]
        + 0.7152 * image[:, :, 1]
        + 0.0722 * image[:, :, 2]
    ).astype(np.float32)


def _ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[2] == 4:
        image = image[:, :, :3]

    return image


def _ensure_mask_uint8(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    if mask.max() <= 1:
        mask = mask * 255

    return np.clip(mask, 0, 255).astype(np.uint8)
