import streamlit as st

from modules.config import configure_page
from modules.session import initialise_session
from modules.sidebar import build_sidebar
from modules.preview import build_preview

configure_page()
initialise_session()

st.title("🎨 Jotun Colour Visualizer")
st.caption("Internal Colour Consultation Tool")

# NEW: show selection status
if st.session_state.selected_surface_point:
    st.success(
        f"Surface seed selected at {st.session_state.selected_surface_point}"
    )

st.divider()

left, right = st.columns([1, 2])

with left:
    build_sidebar()

with right:
    build_preview()
