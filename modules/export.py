from __future__ import annotations

from io import BytesIO
import re

import numpy as np
import streamlit as st
from PIL import Image


def image_array_to_png_bytes(image_np: np.ndarray) -> bytes:
    image = Image.fromarray(image_np.astype("uint8"), mode="RGB")

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return buffer.getvalue()


def build_export_filename(colour: dict | None) -> str:
    if not colour:
        return "jotun_colour_visualizer_export.png"

    colour_code = str(colour.get("colour_code", "colour")).strip()
    colour_name = str(colour.get("colour_name", "preview")).strip()

    raw_name = f"jotun_{colour_code}_{colour_name}.png"

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name)
    safe_name = re.sub(r"_+", "_", safe_name)

    return safe_name.lower()


def render_export_panel(
    painted_image: np.ndarray | None,
    colour: dict | None,
) -> None:
    if painted_image is None:
        return

    st.subheader("Export")

    if colour:
        st.markdown(
            f"""
            **Selected colour:** {colour.get("colour_name", "")}  
            **Colour code:** {colour.get("colour_code", "")}  
            **Product:** {colour.get("product", "")}  
            **Finish:** {colour.get("finish", "")}
            """
        )

    file_name = build_export_filename(colour)

    st.download_button(
        label="Download recoloured image",
        data=image_array_to_png_bytes(painted_image),
        file_name=file_name,
        mime="image/png",
        use_container_width=True,
    )

    st.caption(
        "Digital preview for visual guidance only. Actual colour appearance may vary "
        "depending on lighting, screen calibration, wall texture, surface condition, "
        "paint finish, substrate, and application."
    )
