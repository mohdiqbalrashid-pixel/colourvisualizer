from __future__ import annotations

import cv2
import numpy as np

from modules.config import (
    DETECTION_INITIAL_LAB_TOLERANCE,
    MASK_EXPAND_PIXELS,
    MASK_SHRINK_PIXELS,
    MASK_SMOOTH_KERNEL
)


def _generate_base_mask(
    image: np.ndarray, 
    seed_point: tuple[int, int], 
    tolerance: int
) -> np.ndarray:
    """
    Generates an initial flood-fill mask using LAB colour space for better perceptual matching.
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    height, width = lab_image.shape[:2]
    # FloodFill requires a mask exactly 2 pixels larger than the image
    mask = np.zeros((height + 2, width + 2), np.uint8)
    
    lo_diff = (tolerance, tolerance, tolerance)
    up_diff = (tolerance, tolerance, tolerance)
    
    flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
    
    cv2.floodFill(
        image=lab_image, 
        mask=mask, 
        seedPoint=seed_point, 
        newVal=(255, 255, 255), 
        loDiff=lo_diff, 
        upDiff=up_diff, 
        flags=flags
    )
    
    # Extract the actual mask, removing the +2 padding
    return mask[1:height+1, 1:width+1]


def _refine_mask_with_guided_filter(image: np.ndarray, raw_mask: np.ndarray) -> np.ndarray:
    """
    Refines the mask by snapping to physical edges using a Guided Filter.
    """
    guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Ensure mask is binary before filtering
    _, binary_mask = cv2.threshold(raw_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Guided filter parameters
    radius = MASK_SMOOTH_KERNEL 
    eps = 1e-4 
    
    try:
        # cv2.ximgproc requires opencv-contrib-python
        refined_mask = cv2.ximgproc.guidedFilter(
            guide=guide, 
            src=binary_mask, 
            radius=radius, 
            eps=eps
        )
    except AttributeError:
        # Fallback to standard Gaussian Blur if contrib package is missing
        refined_mask = cv2.GaussianBlur(binary_mask, (radius, radius), 0)
    
    # Morphological adjustments to fill micro-holes
    if MASK_EXPAND_PIXELS > 0:
        kernel = np.ones((MASK_EXPAND_PIXELS, MASK_EXPAND_PIXELS), np.uint8)
        refined_mask = cv2.dilate(refined_mask, kernel, iterations=1)
        
    if MASK_SHRINK_PIXELS > 0:
        kernel = np.ones((MASK_SHRINK_PIXELS, MASK_SHRINK_PIXELS), np.uint8)
        refined_mask = cv2.erode(refined_mask, kernel, iterations=1)
        
    return refined_mask


def create_segmentation_mask(
    image: np.ndarray, 
    seed_point: tuple[int, int], 
    tolerance: int = DETECTION_INITIAL_LAB_TOLERANCE
) -> np.ndarray:
    """
    Main entry point: Generates a raw mask from a click and refines its edges.
    """
    # 1. Ensure seed point is within image bounds
    height, width = image.shape[:2]
    x, y = seed_point
    
    if not (0 <= x < width and 0 <= y < height):
        return np.zeros((height, width), dtype=np.uint8)

    # 2. Generate and refine mask
    raw_mask = _generate_base_mask(image, seed_point, tolerance)
    final_mask = _refine_mask_with_guided_filter(image, raw_mask)
    
    return final_mask
