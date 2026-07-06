import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates


def build_preview():

    st.subheader("Preview")

    if st.session_state.uploaded_image is None:

        st.info("Upload an image to begin.")

        return

    image = st.session_state.uploaded_image

    click = streamlit_image_coordinates(
        image,
        key="image_click",
        use_container_width=True
    )

    if click:

        st.session_state.selected_surface_point = (
            click["x"],
            click["y"]
        )

        st.success(
            f"Selected point → X: {click['x']} | Y: {click['y']}"
        )

    if st.session_state.selected_surface_point:

        st.info(
            f"Active selection: {st.session_state.selected_surface_point}"
        )
