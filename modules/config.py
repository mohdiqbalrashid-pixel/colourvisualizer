import streamlit as st


APP_NAME = "Jotun Colour Visualizer"
APP_VERSION = "v2.1.3"
APP_CAPTION = "Internal Colour Consultation Tool"

PRIMARY_BLUE = "#003E7E"
JOTUN_RED = "#BA0C2F"

MAX_IMAGE_WIDTH = 1200

DEFAULT_MASK_COLOUR = (255, 0, 180)
DEFAULT_MASK_OPACITY = 0.35

DEFAULT_PAINT_STRENGTH = 0.88

MASK_EXPAND_PIXELS = 5
MASK_SHRINK_PIXELS = 5
MASK_SMOOTH_KERNEL = 15

DEFAULT_BRUSH_SIZE = 35
MIN_BRUSH_SIZE = 5
MAX_BRUSH_SIZE = 120

HISTORY_LIMIT = 25


def configure_page() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🎨",
        layout="wide"
    )
