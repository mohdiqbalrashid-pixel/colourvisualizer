from __future__ import annotations

import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from modules.config import (
    DEFAULT_PAINT_STRENGTH,
    MASK_EXPAND_PIXELS,
    MASK_SHRINK_PIXELS,
    MASK_SMOOTH_KERNEL,
)
from modules.mask_editor import (
    clear_editable_mask,
    create_overlay,
    expand_mask,
    fill_mask_holes,
    get_mask_summary,
    has_mask,
    initialise_mask,
    set_editable_mask,
    shrink_mask,
    smooth_mask,
)
from modules.recolouring import apply_paint
from modules.segmentation import create_wall_mask
from modules.session import get_app_state, sync_app_to_legacy


def build_preview() -> None:
    st.subheader("Preview")

    app = get_app_state()

    image = app.get("image") or st.session_state.get("uploaded_image")

    if image is None:
        st.info("Upload an image to begin.")
        return

    image_np = np.array(image)
    initialise_mask(image_np.shape)

    _sync_existing_mask_into_app(image_np)

    _render_mask_controls(image_np)

    display_image = _get_workspace_image(image_np)

    click = streamlit_image_coordinates(
        display_image,
        key="image_click_workspace",
    )

    if click:
        seed = (int(click["x"]), int(click["y"]))

        if seed != app.get("selected_surface_point"):
            _process_surface_click(image_np, seed)
            st.rerun()

    _render_workspace_status(image_np)
    _render_output(image_np)


def _render_mask_controls(image_np: np.ndarray) -> None:
    app = get_app_state()

    controls = st.container()

    with controls:
        col1, col2, col3, col4 = st.columns([1.1, 1, 1, 1])

        with col1:
            show_mask = st.checkbox(
                "Show mask overlay",
                value=bool(app.get("show_mask", True)),
                key="show_mask_overlay_toggle",
            )

            app["show_mask"] = show_mask
            st.session_state.show_mask = show_mask

        mask = app.get("editable_mask")

        with col2:
            if st.button(
                "Expand edge",
                use_container_width=True,
                disabled=not has_mask(mask),
                help="Slightly expands the selected wall mask to reduce unpainted edge gaps.",
            ):
                updated = expand_mask(mask, pixels=MASK_EXPAND_PIXELS)
                _update_mask_and_repaint(image_np, updated)
                st.rerun()

        with col3:
            if st.button(
                "Shrink edge",
                use_container_width=True,
                disabled=not has_mask(mask),
                help="Slightly shrinks the selected wall mask if paint spills outside the wall.",
            ):
                updated = shrink_mask(mask, pixels=MASK_SHRINK_PIXELS)
                _update_mask_and_repaint(image_np, updated)
                st.rerun()

        with col4:
            if st.button(
                "Smooth mask",
                use_container_width=True,
                disabled=not has_mask(mask),
                help="Softens jagged mask boundaries.",
            ):
                updated = smooth_mask(mask, kernel_size=MASK_SMOOTH_KERNEL)
                _update_mask_and_repaint(image_np, updated)
                st.rerun()

        col5, col6 = st.columns([1, 3])

        with col5:
            if st.button(
                "Fill holes",
                use_container_width=True,
                disabled=not has_mask(mask),
                help="Fills small unselected holes inside the wall mask.",
            ):
                updated = fill_mask_holes(mask)
                _update_mask_and_repaint(image_np, updated)
                st.rerun()

        with col6:
            if st.button(
                "Clear mask",
                use_container_width=True,
                disabled=not has_mask(mask),
            ):
                empty = clear_editable_mask(image_np.shape)
                _update_mask_without_repaint(empty)
                app["painted_image"] = None
                st.session_state.painted_image = None
                st.rerun()


def _get_workspace_image(image_np: np.ndarray) -> np.ndarray:
    app = get_app_state()

    mask = app.get("editable_mask")

    if app.get("show_mask", True) and has_mask(mask):
        return create_overlay(image_np, mask)

    return image_np


def _process_surface_click(image_np: np.ndarray, seed: tuple[int, int]) -> None:
    app = get_app_state()

    mask = create_wall_mask(image_np, seed)
    mask = set_editable_mask(mask)

    app["selected_surface_point"] = seed
    app["raw_mask"] = mask.copy()
    app["editable_mask"] = mask.copy()

    st.session_state.selected_surface_point = seed
    st.session_state.wall_mask = mask.copy()
    st.session_state.editable_mask = mask.copy()

    _repaint_from_mask(image_np)

    sync_app_to_legacy()


def _update_mask_and_repaint(image_np: np.ndarray, mask: np.ndarray) -> None:
    app = get_app_state()

    mask = set_editable_mask(mask)

    app["editable_mask"] = mask.copy()
    app["raw_mask"] = mask.copy()

    st.session_state.editable_mask = mask.copy()
    st.session_state.wall_mask = mask.copy()

    _repaint_from_mask(image_np)

    sync_app_to_legacy()


def _update_mask_without_repaint(mask: np.ndarray) -> None:
    app = get_app_state()

    mask = set_editable_mask(mask)

    app["editable_mask"] = mask.copy()
    app["raw_mask"] = mask.copy()

    st.session_state.editable_mask = mask.copy()
    st.session_state.wall_mask = mask.copy()

    sync_app_to_legacy()


def _repaint_from_mask(image_np: np.ndarray) -> None:
    app = get_app_state()

    mask = app.get("editable_mask")
    colour = app.get("selected_colour") or st.session_state.get("selected_colour")

    if mask is None or colour is None or not has_mask(mask):
        app["painted_image"] = None
        st.session_state.painted_image = None
        return

    target_rgb = _colour_to_rgb(colour)

    painted = apply_paint(
        image_np=image_np,
        mask=mask,
        target_rgb=target_rgb,
        strength=DEFAULT_PAINT_STRENGTH,
    )

    app["painted_image"] = painted
    st.session_state.painted_image = painted


def _render_workspace_status(image_np: np.ndarray) -> None:
    app = get_app_state()

    mask = app.get("editable_mask")
    seed = app.get("selected_surface_point")

    if seed is None:
        st.caption("Click directly on the wall or surface you want to recolour.")
    else:
        st.caption(f"Selected surface seed: X={seed[0]}, Y={seed[1]}")

    if has_mask(mask):
        st.caption(get_mask_summary(mask, image_np.shape))


def _render_output(image_np: np.ndarray) -> None:
    app = get_app_state()

    painted = app.get("painted_image")

    if painted is None:
        return

    st.divider()
    st.subheader("Paint Preview")

    before_col, after_col = st.columns(2)

    with before_col:
        st.caption("Before")
        st.image(image_np, use_container_width=True)

    with after_col:
        st.caption("After")
        st.image(painted, use_container_width=True)


def _sync_existing_mask_into_app(image_np: np.ndarray) -> None:
    app = get_app_state()

    legacy_mask = st.session_state.get("editable_mask")
    wall_mask = st.session_state.get("wall_mask")

    if app.get("editable_mask") is None:
        if legacy_mask is not None:
            app["editable_mask"] = legacy_mask.copy()
        elif wall_mask is not None:
            app["editable_mask"] = wall_mask.copy()

    if app.get("raw_mask") is None and wall_mask is not None:
        app["raw_mask"] = wall_mask.copy()

    if app.get("painted_image") is None and st.session_state.get("painted_image") is not None:
        app["painted_image"] = st.session_state.painted_image

    mask = app.get("editable_mask")

    if mask is not None and mask.shape != image_np.shape[:2]:
        app["editable_mask"] = None
        app["raw_mask"] = None
        app["painted_image"] = None
        st.session_state.editable_mask = None
        st.session_state.wall_mask = None
        st.session_state.painted_image = None


def _colour_to_rgb(colour) -> tuple[int, int, int]:
    if hasattr(colour, "to_dict"):
        colour = colour.to_dict()

    return (
        int(colour["r"]),
        int(colour["g"]),
        int(colour["b"]),
    )
