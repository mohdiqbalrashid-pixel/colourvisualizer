from __future__ import annotations

import cv2
import numpy as np

from modules.config import (
    MASK_EXPAND_PIXELS,
    MASK_SHRINK_PIXELS,
    MASK_SMOOTH_KERNEL
)

def _generate_grabcut_mask(
    image: np.ndarray, 
    seed_point: tuple[int, int], 
    radius: int = 50
) -> np.ndarray:
    """
    Uses OpenCV's GrabCut algorithm to intelligently segment the wall based on a click.
    GrabCut handles shadows and textures much better than standard flood-fill.
    """
    height, width = image.shape[:2]
    
    # 1. Initialize empty mask and internal GrabCut arrays
    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # 2. Tell GrabCut that everything is "Probably Background" by default
    mask[:] = cv2.GC_PR_BGD
    
    # 3. Tell GrabCut the area the user clicked is "Definitely Foreground" (the wall)
    x, y = seed_point
    cv2.circle(mask, (x, y), radius, cv2.GC_FGD, -1)
    
    # 4. Optional: Give it a bounding box slightly smaller than the image 
    # to help it understand where the edges of the room might be
    rect = (10, 10, width - 20, height - 20)
    
    # 5. Run GrabCut
    # iterations=3 is a good balance between speed and accuracy for Streamlit
    cv2.grabCut(
        image, 
        mask, 
        rect, 
        bgd_model, 
        fgd_model, 
        3, 
        cv2.GC_INIT_WITH_MASK
    )
    
    # 6. Extract the final binary mask
    # GrabCut marks pixels as 0 (Bg), 1 (Fg), 2 (Prob Bg), 3 (Prob Fg)
    # We want 1 and 3.
    binary_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype("uint8")
    
    return binary_mask

def _refine_mask(image: np.ndarray, raw_mask: np.ndarray) -> np.ndarray:
    """
    Refines edges using Guided Filter to snap perfectly to doorframes/ceilings.
    """
    guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    try:
        refined_mask = cv2.ximgproc.guidedFilter(
            guide=guide, 
            src=raw_mask, 
            radius=MASK_SMOOTH_KERNEL, 
            eps=1e-4
        )
    except AttributeError:
        refined_mask = cv2.GaussianBlur(raw_mask, (MASK_SMOOTH_KERNEL, MASK_SMOOTH_KERNEL), 0)
    
    # Morphological adjustments to fill any remaining micro-holes
    if MASK_EXPAND_PIXELS > 0:
        kernel = np.ones((MASK_EXPAND_PIXELS, MASK_EXPAND_PIXELS), np.uint8)
        refined_mask = cv2.dilate(refined_mask, kernel, iterations=1)
        
    if MASK_SHRINK_PIXELS > 0:
        kernel = np.ones((MASK_SHRINK_PIXELS, MASK_SHRINK_PIXELS), np.uint8)
        refined_mask = cv2.erode(refined_mask, kernel, iterations=1)
        
    return refined_mask

def create_segmentation_mask(
    image: np.ndarray, 
    seed_point: tuple[int, int]
) -> np.ndarray:
    """
    Main entry point. Generates mask using GrabCut and refines it.
    """
    height, width = image.shape[:2]
    x, y = seed_point
    
    if not (0 <= x < width and 0 <= y < height):
        return np.zeros((height, width), dtype=np.uint8)

    raw_mask = _generate_grabcut_mask(image, seed_point)
    final_mask = _refine_mask(image, raw_mask)
    
    return final_mask
