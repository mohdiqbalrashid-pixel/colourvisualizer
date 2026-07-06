import streamlit as st
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

from modules.segmentation import create_wall_mask


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

        st.session_state.selected_surface_point = seed

        mask, segmented = create_wall_mask(image_np, seed)

        st.session_state.wall_mask = mask
        st.session_state.segmented_image = segmented

        st.success(f"Wall detected from seed {seed}")

    # Show results

    if st.session_state.segmented_image is not None:

        col1, col2 = st.columns(2)

        with col1:
            st.write("Original")
            st.image(image_np, use_container_width=True)

        with col2:
            st.write("Detected Wall Region")
            st.image(st.session_state.segmented_image, use_container_width=True)

    else:

        st.image(image_np, use_container_width=True)
