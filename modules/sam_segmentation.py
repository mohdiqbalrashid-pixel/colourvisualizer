import numpy as np
import torch
import cv2

from segment_anything import sam_model_registry, SamPredictor


# Load once (important for performance)
sam = None
predictor = None


def load_sam():

    global sam, predictor

    if sam is not None:
        return predictor

    # You will need to download this checkpoint once
    # https://github.com/facebookresearch/segment-anything
    sam_checkpoint = "sam_vit_b.pth"

    model_type = "vit_b"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    return predictor


def get_sam_mask(image_np, point):

    predictor = load_sam()

    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    predictor.set_image(image_bgr)

    input_point = np.array([point])
    input_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

    # pick best mask
    best_mask = masks[np.argmax(scores)]

    return (best_mask.astype(np.uint8)) * 255
