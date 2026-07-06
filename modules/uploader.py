import streamlit as st

from PIL import Image


def upload_image():

    uploaded = st.file_uploader(
        "Upload customer image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        st.session_state.uploaded_image = image
