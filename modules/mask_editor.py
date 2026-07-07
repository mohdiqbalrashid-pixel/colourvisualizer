from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from modules.config import DEFAULT_MASK_COLOUR, DEFAULT_MASK_OPACITY


def initialise_mask(image_shape: tuple[int, int, int]) -> None:
    h, w = image_shape[:2]

    if (
        "editable_mask" not in st.session_state
        or st.session_state.editable_mask is None
        or st.session_state.editable_mask.shape != (h, w)
    ):
        st.session_state.editable_mask = np.zeros((h, w), dtype=np.uint8)


def set_editable_mask(mask: np.ndarray) -> np.ndarray:
    cleaned = _ensure_uint8_mask(mask)
    st.session_state.editable_mask = cleaned.copy()
    return cleaned


def get_editable_mask() -> np.ndarray | None:
    return st.session_state.get("editable_mask")


def clear_editable_mask(image_shape: tuple[int, int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    empty = np.zeros((h, w), dtype=np.uint8)
    st.session_state.editable_mask = empty
    return empty


def has_mask(mask: np.ndarray | None) -> bool:
    if mask is None:
        return False

    return bool(np.any(mask > 0))


def expand_mask(mask: np.ndarray, pixels: int = 5) -> np.ndarray:
    mask = _ensure_uint8_mask(mask)

    kernel_size = max(3, int(pixels))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    expanded = cv2.dilate(mask, kernel, iterations=1)

    return _ensure_uint8_mask(expanded)


def shrink_mask(mask: np.ndarray, pixels: int = 5) -> np.ndarray:
    mask = _ensure_uint8_mask(mask)

    kernel_size = max(3, int(pixels))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    shrunk = cv2.erode(mask, kernel, iterations=1)

    return _ensure_uint8_mask(shrunk)


def smooth_mask(mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    mask = _ensure_uint8_mask(mask)

    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    smoothed = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    return _ensure_uint8_mask(smoothed)


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    mask = _ensure_uint8_mask(mask)

    binary = (mask > 0).astype(np.uint8) * 255

    h, w = binary.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    flood_filled = binary.copy()
    cv2.floodFill(flood_filled, flood_mask, (0, 0), 255)

    inverted = cv2.bitwise_not(flood_filled)
    filled = binary | inverted

    return _ensure_uint8_mask(filled)


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    colour: tuple[int, int, int] = DEFAULT_MASK_COLOUR,
    opacity: float = DEFAULT_MASK_OPACITY,
) -> np.ndarray:
    if mask is None:
        mask = get_editable_mask()

    if mask is None or not has_mask(mask):
        return image.copy()

    mask = _ensure_uint8_mask(mask)

    overlay = image.copy()
    mask_bool = mask > 0

    colour_layer = np.zeros_like(image)
    colour_layer[:, :] = colour

    blended = cv2.addWeighted(
        image,
        1 - opacity,
        colour_layer,
        opacity,
        0,
    )

    overlay[mask_bool] = blended[mask_bool]

    return overlay


def get_mask_coverage(mask: np.ndarray | None, image_shape: tuple[int, int, int]) -> float:
    if mask is None:
        return 0.0

    h, w = image_shape[:2]
    total_pixels = h * w

    if total_pixels == 0:
        return 0.0

    selected_pixels = int(np.sum(mask > 0))

    return selected_pixels / total_pixels


def get_mask_summary(mask: np.ndarray | None, image_shape: tuple[int, int, int]) -> str:
    if mask is None or not has_mask(mask):
        return "No mask selected."

    coverage = get_mask_coverage(mask, image_shape)
    selected_pixels = int(np.sum(mask > 0))

    return f"Mask coverage: {coverage:.1%} | Selected pixels: {selected_pixels:,}"


def _ensure_uint8_mask(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        raise ValueError("Mask cannot be None.")

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    if mask.max() <= 1:
        mask = mask * 255

    return np.clip(mask, 0, 255).astype(np.uint8)
