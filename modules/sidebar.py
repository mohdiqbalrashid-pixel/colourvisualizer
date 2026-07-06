import streamlit as st

from modules.uploader import upload_image
from modules.colour_database import load_colours


def build_sidebar():

    st.subheader("Project")

    upload_image()

    st.divider()

    st.subheader("Colour Library")

    colours = load_colours()

    search = st.text_input(
        "Search",
        placeholder="Colour name or code..."
    )

    if search:

        colours = colours[
            colours["colour_name"].str.contains(
                search,
                case=False,
                na=False
            )
            |
            colours["colour_code"].astype(str).str.contains(search)
        ]

    if colours.empty:

        st.warning("No colours found.")
        return

    labels = (
        colours["colour_code"].astype(str)
        + " • "
        + colours["colour_name"]
    )

    selected = st.selectbox(
        "Available Colours",
        labels
    )

    row = colours.iloc[labels.tolist().index(selected)]

    st.session_state.selected_colour = row

    st.markdown("### Selected Colour")

    st.markdown(
        f"""
<div style="
height:70px;
border-radius:8px;
background:{row['hex']};
border:1px solid #cccccc;">
</div>
""",
        unsafe_allow_html=True
    )

    st.write(f"**Name:** {row['colour_name']}")
    st.write(f"**Code:** {row['colour_code']}")
    st.write(f"**Product:** {row['product']}")
    st.write(f"**Finish:** {row['finish']}")
