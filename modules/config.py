import streamlit as st


APP_NAME = "Jotun Colour Visualizer"
APP_VERSION = "v2.1.8"
APP_CAPTION = "Internal Colour Consultation Tool"

PRIMARY_BLUE = "#003E7E"
JOTUN_RED = "#BA0C2F"

MAX_IMAGE_WIDTH = 1200

DEFAULT_MASK_COLOUR = (255, 0, 180)
DEFAULT_MASK_OPACITY = 0.35

# Set to 1.0 so the painted colour matches the selected RGB as closely as possible
# while the new recolouring engine still preserves shadows and highlights.
DEFAULT_PAINT_STRENGTH = 1.0

MASK_EXPAND_PIXELS = 5
MASK_SHRINK_PIXELS = 5
MASK_SMOOTH_KERNEL = 15

DEFAULT_BRUSH_SIZE = 35
MIN_BRUSH_SIZE = 5
MAX_BRUSH_SIZE = 120

HISTORY_LIMIT = 25

DEFAULT_COMPARISON_POSITION = 50

DETECTION_MIN_COVERAGE = 0.003
DETECTION_MAX_COVERAGE = 0.48
DETECTION_EDGE_LOW = 45
DETECTION_EDGE_HIGH = 135
DETECTION_INITIAL_LAB_TOLERANCE = 24
DETECTION_MIN_LAB_TOLERANCE = 10
DETECTION_MAX_LAB_TOLERANCE = 42


def configure_page() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🎨",
        layout="wide"
    )
