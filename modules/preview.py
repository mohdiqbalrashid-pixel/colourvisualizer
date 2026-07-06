import streamlit as st
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

from modules.sam_segmentation import get_sam_mask
from modules.recolouring import apply_paint


def build_preview():

    st.subheader("Preview")

    if st.session_state.uploaded_image is None:
        st.info("Upload an image to begin.")
        return

    image_np = np.array(st.session_state.uploaded_image)

    click = streamlit_image_coordinates(
        image_np,
        key="image_click"
    )

    if click:

        seed = (click["x"], click["y"])

        if seed != st.session_state.selected_surface_point:

            st.session_state.selected_surface_point = seed

            # 🔥 SAM INFERENCE
            mask = get_sam_mask(image_np, seed)

            st.session_state.wall_mask = mask

            if st.session_state.selected_colour is not None:

                painted = apply_paint(
                    image_np,
                    mask,
                    (
                        st.session_state.selected_colour["r"],
                        st.session_state.selected_colour["g"],
                        st.session_state.selected_colour["b"]
                    )
                )

                st.session_state.painted_image = painted

            st.success(f"AI wall detected from {seed}")

    if st.session_state.painted_image is not None:

        col1, col2 = st.columns(2)

        with col1:
            st.image(image_np, use_container_width=True)

        with col2:
            st.image(st.session_state.painted_image, use_container_width=True)

    else:
        st.image(image_np, use_container_width=True)
