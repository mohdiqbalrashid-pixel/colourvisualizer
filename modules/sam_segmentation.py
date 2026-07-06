import streamlit as st
import numpy as np
import torch
import cv2
import os
import urllib.request

from segment_anything import sam_model_registry, SamPredictor


SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
MODEL_PATH = "sam_vit_b_01ec64.pth"

sam = None
predictor = None


def download_model():

    if os.path.exists(MODEL_PATH):
        return

    with st.spinner("Downloading SAM model (first run only)..."):
        urllib.request.urlretrieve(SAM_URL, MODEL_PATH)


def load_sam():

    global sam, predictor

    if predictor is not None:
        return predictor

    download_model()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry["vit_b"](checkpoint=MODEL_PATH)
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

    best_mask = masks[np.argmax(scores)]

    return (best_mask.astype(np.uint8)) * 255
