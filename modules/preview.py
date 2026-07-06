import streamlit as st


def build_preview():

    st.subheader("Preview")

    if st.session_state.uploaded_image is None:

        st.info("Upload an image to begin.")

        return

    st.image(
        st.session_state.uploaded_image,
        use_container_width=True
    )
