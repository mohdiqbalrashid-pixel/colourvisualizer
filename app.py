import streamlit as st
import pandas as pd
from PIL import Image

# ---------------------------------------------------
# Page setup
# ---------------------------------------------------

st.set_page_config(
if "selected_point" not in st.session_state:
    st.session_state["selected_point"] = None
    page_title="Jotun Colour Visualizer",
    page_icon="🎨",
    layout="wide"
)

# ---------------------------------------------------
# Load colours
# ---------------------------------------------------

@st.cache_data
def load_colours():
    return pd.read_csv("colours.csv")

colours = load_colours()

# ---------------------------------------------------
# Header
# ---------------------------------------------------

st.title("🎨 Jotun Colour Visualizer")

st.caption(
    "Internal colour consultation tool"
)

st.divider()

# ---------------------------------------------------
# Main layout
# ---------------------------------------------------

left, right = st.columns([1, 2])

# ===================================================
# LEFT PANEL
# ===================================================

with left:

    st.subheader("Project")

    uploaded_file = st.file_uploader(
        "Upload room photo",
        type=["jpg", "jpeg", "png"]
    )

    st.divider()

    st.subheader("Colour Selection")

    search = st.text_input(
        "Search colour",
        placeholder="Example: Timeless or 1024"
    )

    if search:

        filtered = colours[
            colours["colour_name"].str.contains(search, case=False)
            |
            colours["colour_code"].astype(str).str.contains(search)
        ]

    else:

        filtered = colours

    labels = (
        filtered["colour_code"].astype(str)
        + " • "
        + filtered["colour_name"]
    )

    selected = st.selectbox(
        "Available colours",
        labels
    )

    row = filtered.iloc[labels.tolist().index(selected)]

    st.markdown("### Selected Colour")

    swatch_html = f"""
    <div style="
        width:100%;
        height:70px;
        border-radius:10px;
        background:{row['hex']};
        border:1px solid #CCCCCC;">
    </div>
    """

    st.markdown(
        swatch_html,
        unsafe_allow_html=True
    )

    st.write(f"**Name:** {row['colour_name']}")
    st.write(f"**Code:** {row['colour_code']}")
    st.write(f"**HEX:** {row['hex']}")
    st.write(f"**Product:** {row['product']}")
    st.write(f"**Finish:** {row['finish']}")

    st.divider()

    st.button(
        "Apply Colour",
        type="primary",
        use_container_width=True
    )

# ===================================================
# RIGHT PANEL
# ===================================================

with right:

    st.subheader("Preview")

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        click = streamlit_image_coordinates(
            image,
            key="room_image",
            use_container_width=True
        )

        if click:

            st.success(
                f"Clicked at X={click['x']}  Y={click['y']}"
            )

            st.session_state["selected_point"] = (
                click["x"],
                click["y"]
            )

    else:

        st.info("Upload an image to begin.")

from streamlit_image_coordinates import streamlit_image_coordinates
