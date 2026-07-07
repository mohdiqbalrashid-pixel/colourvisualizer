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
    Colour-accurate paint engine v3.

    This version is designed for visualisation where the selected colour should
    drive the result, not the existing wall colour.

    It works by:
    - converting the target RGB to LAB
    - using target LAB as the base colour
    - preserving only brightness variation from the original wall
    - ignoring the original wall chroma / hue
    - keeping sunlight, shadows and texture via the L channel
    """

    if image_np is None or mask is None:
        return image_np

    image_uint8 = _ensure_rgb_uint8(image_np)
    mask_uint8 = _ensure_mask_uint8(mask)

    alpha = _create_boundary_alpha(mask_uint8, strength)

    if not np.any(alpha > 0):
        return image_uint8

    image_lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)

    target_rgb_array = np.uint8([[target_rgb]])
    target_lab = cv2.cvtColor(target_rgb_array, cv2.COLOR_RGB2LAB)[0][0].astype(np.float32)

    original_l = image_lab[:, :, 0]

    new_l = _build_target_lightness(
        original_lightness=original_l,
        mask=mask_uint8,
        target_lightness=float(target_lab[0]),
    )

    painted_lab = np.zeros_like(image_lab)
    painted_lab[:, :, 0] = new_l
    painted_lab[:, :, 1] = float(target_lab[1])
    painted_lab[:, :, 2] = float(target_lab[2])

    painted_lab = np.clip(painted_lab, 0, 255).astype(np.uint8)
    painted_rgb = cv2.cvtColor(painted_lab, cv2.COLOR_LAB2RGB).astype(np.float32)

    original_rgb = image_uint8.astype(np.float32)

    final = (
        original_rgb * (1.0 - alpha[:, :, None])
        + painted_rgb * alpha[:, :, None]
    )

    return np.clip(final, 0, 255).astype(np.uint8)


def _build_target_lightness(
    original_lightness: np.ndarray,
    mask: np.ndarray,
    target_lightness: float,
) -> np.ndarray:
    """
    Preserve lighting without preserving original colour.

    The selected RGB becomes the base colour.
    The original wall only contributes brightness variation:
    - shadows stay darker
    - sunlit areas stay brighter
    - texture stays visible
    """

    mask_bool = mask > 0

    if not np.any(mask_bool):
        return np.full_like(original_lightness, target_lightness, dtype=np.float32)

    selected_l = original_lightness[mask_bool]

    # Reference point for "normal wall brightness".
    # Using median prevents the existing wall colour from overpowering the selected paint.
    reference_l = float(np.percentile(selected_l, 55))
    reference_l = max(reference_l, 1.0)

    broad_delta = original_lightness - reference_l

    # Large-scale lighting: sunlight and shadow.
    lighting_gain = 0.72
    lighting_component = broad_delta * lighting_gain

    # Fine wall texture.
    blurred_l = cv2.GaussianBlur(original_lightness, (0, 0), sigmaX=7, sigmaY=7)
    texture_delta = original_lightness - blurred_l

    texture_gain = 0.25
    texture_component = texture_delta * texture_gain

    new_l = target_lightness + lighting_component + texture_component

    # Avoid destroying the selected colour in very dark or very bright zones.
    lower_bound = max(0, target_lightness - 70)
    upper_bound = min(255, target_lightness + 75)

    new_l = np.clip(new_l, lower_bound, upper_bound)

    return new_l.astype(np.float32)


def _create_boundary_alpha(mask: np.ndarray, strength: float) -> np.ndarray:
    """
    Strong interior coverage, soft boundary.

    Interior pixels should strongly match the selected colour.
    Boundary pixels blend softly to avoid harsh cut lines.
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

    soft = cv2.GaussianBlur(base, (17, 17), 0)
    soft = np.clip(soft, 0.0, 1.0)

    alpha = (
        0.42 * soft
        + 0.48 * binary.astype(np.float32)
        + 0.10 * edge_ring
    )

    # Force the interior to behave like actual selected paint.
    alpha = np.maximum(alpha, core * 0.995)

    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    alpha = np.clip(alpha, 0.0, 1.0)

    return alpha * float(strength)


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
