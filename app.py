import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(
    page_title="Jotun Colour Visualizer",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Jotun Colour Visualizer")

st.write("Welcome! Upload a room photo to begin.")

# Load colours
colours = pd.read_csv("colours.csv")

st.success(f"Loaded {len(colours)} colours.")

uploaded_file = st.file_uploader(
    "Upload a room image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )
