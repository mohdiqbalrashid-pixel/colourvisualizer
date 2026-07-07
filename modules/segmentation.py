from __future__ import annotations

import cv2
import numpy as np

from modules.config import (
    DETECTION_EDGE_HIGH,
    DETECTION_EDGE_LOW,
    DETECTION_INITIAL_LAB_TOLERANCE,
    DETECTION_MAX_COVERAGE,
    DETECTION_MAX_LAB_TOLERANCE,
    DETECTION_MIN_COVERAGE,
    DETECTION_MIN_LAB_TOLERANCE,
)


def create_wall_mask(image_np: np.ndarray, seed_point: tuple[int, int]) -> np.ndarray:
    """
    Create a first-pass editable mask from a clicked wall point.

    This is a stable no-AI segmentation engine designed for Streamlit Cloud.

    It combines:
    - LAB colour similarity
    - edge barriers
    - connected component selection
    - coverage safety limits
    - morphology cleanup

    The goal is not to be perfect. The goal is to create a good first mask
    that the consultant can quickly correct with Brush / Eraser.
    """

    image_np = _ensure_rgb_uint8(image_np)

    height, width = image_np.shape[:2]
    x, y = _clamp_seed(seed_point, width, height)

    edges = _create_edge_barrier(image_np)
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)

    seed_lab = _get_seed_lab_average(lab, x, y)

    mask = _find_best_region(
        lab=lab,
        seed_lab=seed_lab,
        seed=(x, y),
        edges=edges,
        image_shape=(height, width),
    )

    mask = _clean_mask(mask)

    mask = _safety_trim_if_needed(mask, seed=(x, y), image_shape=(height, width))

    return mask


def _find_best_region(
    lab: np.ndarray,
    seed_lab: np.ndarray,
    seed: tuple[int, int],
    edges: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    height, width = image_shape

    tolerances = _build_tolerance_sequence()

    best_mask = np.zeros((height, width), dtype=np.uint8)
    best_score = -1.0

    for tolerance in tolerances:
        candidate = _create_colour_candidate(
            lab=lab,
            seed_lab=seed_lab,
            tolerance=tolerance,
            edges=edges,
            seed=seed,
        )

        mask = _flood_candidate_from_seed(candidate, seed)

        coverage = _coverage(mask)

        score = _score_mask(coverage)

        if score > best_score:
            best_score = score
            best_mask = mask

        if DETECTION_MIN_COVERAGE <= coverage <= DETECTION_MAX_COVERAGE:
            return mask

    return best_mask


def _build_tolerance_sequence() -> list[int]:
    """
    Try tighter and looser LAB thresholds.

    We start around the default tolerance, then explore both directions.
    This avoids full-image masks on plain photos while still allowing large walls.
    """

    initial = int(DETECTION_INITIAL_LAB_TOLERANCE)

    sequence = [
        initial,
        initial - 4,
        initial - 8,
        initial - 12,
        initial + 4,
        initial + 8,
        initial + 12,
        initial + 18,
    ]

    cleaned = []

    for value in sequence:
        value = int(np.clip(
            value,
            DETECTION_MIN_LAB_TOLERANCE,
            DETECTION_MAX_LAB_TOLERANCE,
        ))

        if value not in cleaned:
            cleaned.append(value)

    return cleaned


def _create_colour_candidate(
    lab: np.ndarray,
    seed_lab: np.ndarray,
    tolerance: int,
    edges: np.ndarray,
    seed: tuple[int, int],
) -> np.ndarray:
    distance = np.linalg.norm(
        lab.astype(np.float32) - seed_lab.astype(np.float32),
        axis=2,
    )

    colour_similar = distance <= float(tolerance)

    edge_blocked = edges > 0

    candidate = np.logical_and(colour_similar, ~edge_blocked).astype(np.uint8)

    x, y = seed
    candidate[y, x] = 1

    return candidate


def _flood_candidate_from_seed(
    candidate: np.ndarray,
    seed: tuple[int, int],
) -> np.ndarray:
    height, width = candidate.shape[:2]

    flood_source = (candidate * 255).astype(np.uint8)

    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

    x, y = seed

    cv2.floodFill(
        flood_source,
        flood_mask,
        seedPoint=(x, y),
        newVal=128,
        loDiff=0,
        upDiff=0,
        flags=4,
    )

    selected = (flood_source == 128).astype(np.uint8) * 255

    return selected


def _create_edge_barrier(image_np: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(
        blurred,
        threshold1=DETECTION_EDGE_LOW,
        threshold2=DETECTION_EDGE_HIGH,
    )

    # Strengthen architectural boundaries so flood regions do not leak.
    kernel = np.ones((3, 3), dtype=np.uint8)

    edges = cv2.dilate(edges, kernel, iterations=1)

    # Add a second pass of softer edges based on gradients.
    sobel_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)

    gradient = cv2.magnitude(sobel_x, sobel_y)
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, gradient_edges = cv2.threshold(
        gradient,
        55,
        255,
        cv2.THRESH_BINARY,
    )

    gradient_edges = cv2.dilate(gradient_edges, kernel, iterations=1)

    combined = cv2.bitwise_or(edges, gradient_edges)

    return combined


def _get_seed_lab_average(lab: np.ndarray, x: int, y: int) -> np.ndarray:
    height, width = lab.shape[:2]

    radius = 3

    x1 = max(0, x - radius)
    x2 = min(width, x + radius + 1)

    y1 = max(0, y - radius)
    y2 = min(height, y + radius + 1)

    patch = lab[y1:y2, x1:x2]

    return patch.reshape(-1, 3).mean(axis=0)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = _ensure_mask_uint8(mask)

    binary = (mask > 0).astype(np.uint8) * 255

    kernel_small = np.ones((3, 3), dtype=np.uint8)
    kernel_medium = np.ones((5, 5), dtype=np.uint8)

    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_small, iterations=1)

    cleaned = _keep_largest_component(cleaned)

    cleaned = _fill_internal_holes(cleaned)

    return _ensure_mask_uint8(cleaned)


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]

    if len(areas) == 0:
        return mask

    largest_label = int(np.argmax(areas)) + 1

    largest = (labels == largest_label).astype(np.uint8) * 255

    return largest


def _fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255

    height, width = binary.shape[:2]
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

    flood = binary.copy()
    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    holes = cv2.bitwise_not(flood)

    filled = cv2.bitwise_or(binary, holes)

    return filled


def _safety_trim_if_needed(
    mask: np.ndarray,
    seed: tuple[int, int],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """
    If the detected region is still too large, apply a conservative local
    limiter around the clicked point.

    This prevents the mask from becoming "the entire photograph".
    """

    coverage = _coverage(mask)

    if coverage <= DETECTION_MAX_COVERAGE:
        return mask

    height, width = image_shape
    x, y = seed

    limited = np.zeros((height, width), dtype=np.uint8)

    # Generous local box. Large enough for a wall, small enough to prevent
    # full-image takeover.
    box_half_width = int(width * 0.38)
    box_half_height = int(height * 0.38)

    x1 = max(0, x - box_half_width)
    x2 = min(width, x + box_half_width)

    y1 = max(0, y - box_half_height)
    y2 = min(height, y + box_half_height)

    limited[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

    limited = _clean_mask(limited)

    return limited


def _score_mask(coverage: float) -> float:
    """
    Prefer medium-sized wall-like regions.

    Too tiny = not useful.
    Too huge = probably leaked.
    """

    if coverage <= 0:
        return -1.0

    if DETECTION_MIN_COVERAGE <= coverage <= DETECTION_MAX_COVERAGE:
        # Prefer regions around 10-35%.
        target = 0.22
        return 1.0 - abs(coverage - target)

    if coverage < DETECTION_MIN_COVERAGE:
        return coverage

    return max(0.0, 1.0 - coverage)


def _coverage(mask: np.ndarray) -> float:
    if mask is None or mask.size == 0:
        return 0.0

    return float(np.sum(mask > 0) / mask.size)


def _clamp_seed(
    seed_point: tuple[int, int],
    width: int,
    height: int,
) -> tuple[int, int]:
    x, y = seed_point

    x = int(np.clip(x, 0, width - 1))
    y = int(np.clip(y, 0, height - 1))

    return x, y


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
