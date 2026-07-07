from __future__ import annotations

import streamlit as st
from PIL import Image, ImageOps

from modules.config import MAX_IMAGE_WIDTH
from modules.session import get_app_state, sync_app_to_legacy


def _load_image(uploaded_file) -> Image.Image:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def _resize_image(image: Image.Image, max_width: int = MAX_IMAGE_WIDTH) -> Image.Image:
    width, height = image.size

    if width <= max_width:
        return image

    scale = max_width / width
    new_height = int(height * scale)

    return image.resize((max_width, new_height))


def _clear_image_outputs() -> None:
    app = get_app_state()

    app["selected_surface_point"] = None
    app["raw_mask"] = None
    app["editable_mask"] = None
    app["painted_image"] = None
    app["history"] = []
    app["redo_stack"] = []

    sync_app_to_legacy()


def upload_image() -> None:
    app = get_app_state()

    uploaded_file = st.file_uploader(
        "Upload customer image",
        type=["jpg", "jpeg", "png"],
        key="customer_image_upload",
    )

    if uploaded_file is None:
        if app.get("image") is None:
            st.caption("No image uploaded yet.")
        return

    uploaded_signature = f"{uploaded_file.name}_{uploaded_file.size}"

    if app.get("image_name") != uploaded_signature:
        image = _load_image(uploaded_file)
        image = _resize_image(image)

        app["image"] = image
        app["image_name"] = uploaded_signature

        _clear_image_outputs()

        sync_app_to_legacy()

    image = app.get("image")

    if image is not None:
        width, height = image.size

        st.caption(
            f"Loaded: **{uploaded_file.name}**  \n"
            f"Working size: **{width} × {height}px**"
        )
