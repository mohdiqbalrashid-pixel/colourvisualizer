from __future__ import annotations

import cv2
import numpy as np

from modules.config import DEFAULT_PAINT_STRENGTH


def apply_realistic_paint(
    original_image: np.ndarray, 
    mask: np.ndarray, 
    target_rgb: tuple[int, int, int], 
    paint_strength: float = DEFAULT_PAINT_STRENGTH
) -> np.ndarray:
    """
    Applies a new colour to a masked area using CIELAB colour space.
    Maintains the original texture and lighting while adjusting luminance naturally.
    """
    # Ensure mask is 2D
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
    # 1. Convert original image to LAB
    lab_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # 2. Convert target Jotun RGB to LAB
    target_img = np.uint8([[target_rgb]])
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB)[0][0].astype(np.float32)
    target_l, target_a, target_b = target_lab
    
    # Create boolean mask for processing
    bool_mask = mask > 0
    
    # Failsafe: return original if mask is empty
    if not np.any(bool_mask):
        return original_image.copy()
        
    # 3. Calculate average Lightness of the original masked area
    masked_pixels = l_channel[bool_mask]
    avg_original_l = np.mean(masked_pixels)
    
    # 4. Calculate Luminance shift (difference in brightness)
    l_shift = target_l - avg_original_l
    
    # 5. Apply the new colour to the masked area
    new_l = np.clip(l_channel + l_shift, 0, 255)
    
    l_channel[bool_mask] = new_l[bool_mask]
    
    # Apply color channels, respecting paint strength (usually 1.0 for full coverage)
    a_channel[bool_mask] = (target_a * paint_strength) + (a_channel[bool_mask] * (1 - paint_strength))
    b_channel[bool_mask] = (target_b * paint_strength) + (b_channel[bool_mask] * (1 - paint_strength))
    
    # 6. Merge and convert back to RGB
    merged_lab = cv2.merge([l_channel, a_channel, b_channel]).astype(np.uint8)
    final_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    
    # 7. Blend the edges smoothly using the original mask's gradients/anti-aliasing
    mask_normalized = mask.astype(np.float32) / 255.0
    mask_3d = np.dstack([mask_normalized] * 3)
    
    result = (final_rgb * mask_3d + original_image * (1 - mask_3d)).astype(np.uint8)
    
    return result
