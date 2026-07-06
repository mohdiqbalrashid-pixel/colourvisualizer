import streamlit as st

from modules.config import configure_page
from modules.session import initialise_session
from modules.sidebar import build_sidebar
from modules.preview import build_preview

# -------------------------------------------------
# Configure page
# -------------------------------------------------

configure_page()

# -------------------------------------------------
# Session State
# -------------------------------------------------

initialise_session()

# -------------------------------------------------
# Page Header
# -------------------------------------------------

st.title("🎨 Jotun Colour Visualizer")
st.caption("Internal Colour Consultation Tool")

st.divider()

# -------------------------------------------------
# Layout
# -------------------------------------------------

left, right = st.columns([1, 2])

with left:
    build_sidebar()

with right:
    build_preview()
