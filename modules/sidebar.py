from __future__ import annotations

import streamlit as st

from modules.colour_database import load_colours
from modules.session import get_app_state, sync_app_to_legacy
from modules.uploader import upload_image


def _render_colour_swatch(hex_value: str) -> None:
    st.markdown(
        f"""
        <div style="
            height:76px;
            border-radius:10px;
            background:{hex_value};
            border:1px solid #CFCFCF;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.25);
        ">
        </div>
        """,
        unsafe_allow_html=True,
    )


def _filter_colours(search_text: str):
    colours = load_colours()

    if not search_text:
        return colours

    search_text = search_text.strip()

    return colours[
        colours["colour_name"].str.contains(search_text, case=False, na=False)
        |
        colours["colour_code"].astype(str).str.contains(search_text, case=False, na=False)
        |
        colours["product"].str.contains(search_text, case=False, na=False)
        |
        colours["finish"].str.contains(search_text, case=False, na=False)
    ]


def _select_colour(filtered_colours):
    app = get_app_state()

    labels = filtered_colours["display_label"].tolist()

    selected_label = st.selectbox(
        "Available colours",
        labels,
        key="colour_select",
    )

    selected_row = filtered_colours[
        filtered_colours["display_label"] == selected_label
    ].iloc[0]

    colour = selected_row.to_dict()

    app["selected_colour"] = colour
    sync_app_to_legacy()

    return colour


def build_sidebar() -> None:
    app = get_app_state()

    st.subheader("Project")
    upload_image()

    if app.get("image") is not None:
        if st.button("Reset mask and preview", use_container_width=True):
            app["selected_surface_point"] = None
            app["raw_mask"] = None
            app["editable_mask"] = None
            app["painted_image"] = None
            app["history"] = []
            app["redo_stack"] = []

            sync_app_to_legacy()
            st.rerun()

    st.divider()

    st.subheader("Colour Library")

    search = st.text_input(
        "Search",
        placeholder="Colour name, code, product or finish...",
        key="colour_search",
    )

    filtered_colours = _filter_colours(search)

    if filtered_colours.empty:
        st.warning("No colours found.")
        return

    colour = _select_colour(filtered_colours)

    st.markdown("### Selected Colour")

    _render_colour_swatch(colour["hex"])

    st.write(f"**Name:** {colour['colour_name']}")
    st.write(f"**Code:** {colour['colour_code']}")
    st.write(f"**HEX:** `{colour['hex']}`")
    st.write(f"**Product:** {colour['product']}")
    st.write(f"**Finish:** {colour['finish']}")

    if str(colour.get("lrv", "")).strip():
        st.write(f"**LRV:** {colour['lrv']}")

    st.caption(f"{len(filtered_colours)} colour(s) available in current search.")
