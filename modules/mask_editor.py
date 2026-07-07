import streamlit as st
import numpy as np
import cv2


def initialise_mask(image_shape):

    if (
        "editable_mask" not in st.session_state
        or st.session_state.editable_mask is None
    ):

        h, w = image_shape[:2]

        st.session_state.editable_mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )


def load_mask(mask):

    st.session_state.editable_mask = mask.copy()


def get_mask():

    return st.session_state.editable_mask


def clear_mask():

    if st.session_state.editable_mask is not None:

        st.session_state.editable_mask[:] = 0


def create_overlay(image, colour=(255, 0, 180), opacity=0.35):

    if st.session_state.editable_mask is None:

        return image

    overlay = image.copy()

    mask = st.session_state.editable_mask > 0

    colour_layer = np.zeros_like(image)

    colour_layer[:, :] = colour

    overlay[mask] = cv2.addWeighted(
        image,
        1 - opacity,
        colour_layer,
        opacity,
        0
    )[mask]

    return overlay
