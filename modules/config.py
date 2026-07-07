import streamlit as st


APP_NAME = "Jotun Colour Visualizer"
APP_VERSION = "v2.1.0"
APP_CAPTION = "Internal Colour Consultation Tool"

PRIMARY_BLUE = "#003E7E"
JOTUN_RED = "#BA0C2F"

MAX_IMAGE_WIDTH = 1200

DEFAULT_MASK_COLOUR = (255, 0, 180)
DEFAULT_MASK_OPACITY = 0.35

DEFAULT_PAINT_STRENGTH = 0.85


def configure_page() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🎨",
        layout="wide"
    )
