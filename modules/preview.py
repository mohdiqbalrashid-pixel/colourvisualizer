from __future__ import annotations

import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from modules.config import (
    DEFAULT_BRUSH_SIZE,
    DEFAULT_PAINT_STRENGTH,
    MASK_EXPAND_PIXELS,
    MASK_SHRINK_PIXELS,
    MASK_SMOOTH_KERNEL,
    MAX_BRUSH_SIZE,
    MIN_BRUSH_SIZE,
)
from modules.export import render_export_panel
from modules.mask_editor import (
    apply_brush_to_mask,
    can_redo,
    can_undo,
    clear_editable_mask,
    create_overlay,
    expand_mask,
    fill_mask_holes,
    get_mask_summary,
    has_mask,
    initialise_mask,
    push_history,
    redo_mask,
    set_editable_mask,
    shrink_mask,
    smooth_mask,
    undo_mask,
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
    _maybe_repaint_for_colour_change(image_np)

    _render_mask_controls(image_np)

    display_image = _get_workspace_image(image_np)

    click = streamlit_image_coordinates(
        display_image,
        key="image_click_workspace",
    )

    if click:
        point = (int(click["x"]), int(click["y"]))
        _handle_workspace_click(image_np, point, click)

    _render_workspace_status(image_np)
    _render_output(image_np)


def _render_mask_controls(image_np: np.ndarray) -> None:
    app = get_app_state()

    st.markdown("#### Surface Tools")

    tool_labels = {
        "Select / Replace": "select",
        "Add Surface": "add_surface",
        "Remove Surface": "remove_surface",
        "Brush Add": "brush",
        "Eraser": "erase",
    }

    current_tool = app.get("active_tool", "select")

    reverse_lookup = {value: label for label, value in tool_labels.items()}
    current_label = reverse_lookup.get(current_tool, "Select / Replace")

    tool_options = list(tool_labels.keys())

    top_col1, top_col2, top_col3 = st.columns([1, 2.1, 1])

    with top_col1:
        show_mask = st.checkbox(
            "Show mask",
            value=bool(app.get("show_mask", True)),
            key="show_mask_overlay_toggle",
        )

        app["show_mask"] = show_mask
        st.session_state.show_mask = show_mask

    with top_col2:
        selected_tool_label = st.radio(
            "Active tool",
            tool_options,
            index=tool_options.index(current_label),
            horizontal=True,
            key="mask_tool_radio",
        )

        app["active_tool"] = tool_labels[selected_tool_label]
        st.session_state.active_tool = app["active_tool"]

    with top_col3:
        brush_size = st.slider(
            "Brush size",
            min_value=MIN_BRUSH_SIZE,
            max_value=MAX_BRUSH_SIZE,
            value=int(app.get("brush_size", DEFAULT_BRUSH_SIZE)),
            step=5,
            key="brush_size_slider",
        )

        app["brush_size"] = brush_size
        st.session_state.brush_size = brush_size

    st.caption(
        "**Select / Replace** starts a new mask. "
        "**Add Surface** adds another detected area. "
        "**Remove Surface** subtracts a detected area. "
        "Brush and Eraser are for small final touch-ups."
    )

    mask = app.get("editable_mask")

    row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)

    with row1_col1:
        if st.button(
            "Expand edge",
            use_container_width=True,
            disabled=not has_mask(mask),
            help="Slightly expands the selected mask to reduce unpainted edge gaps.",
        ):
            updated = expand_mask(mask, pixels=MASK_EXPAND_PIXELS)
            _update_mask_with_history_and_repaint(image_np, updated)
            st.rerun()

    with row1_col2:
        if st.button(
            "Shrink edge",
            use_container_width=True,
            disabled=not has_mask(mask),
            help="Slightly shrinks the selected mask if paint spills outside the wall.",
        ):
            updated = shrink_mask(mask, pixels=MASK_SHRINK_PIXELS)
            _update_mask_with_history_and_repaint(image_np, updated)
            st.rerun()

    with row1_col3:
        if st.button(
            "Smooth mask",
            use_container_width=True,
            disabled=not has_mask(mask),
            help="Softens jagged mask boundaries.",
        ):
            updated = smooth_mask(mask, kernel_size=MASK_SMOOTH_KERNEL)
            _update_mask_with_history_and_repaint(image_np, updated)
            st.rerun()

    with row1_col4:
        if st.button(
            "Fill holes",
            use_container_width=True,
            disabled=not has_mask(mask),
            help="Fills small unselected holes inside the wall mask.",
        ):
            updated = fill_mask_holes(mask)
            _update_mask_with_history_and_repaint(image_np, updated)
            st.rerun()

    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row2_col1:
        if st.button(
            "Undo",
            use_container_width=True,
            disabled=not can_undo(),
        ):
            updated = undo_mask(mask)
            if updated is not None:
                _update_mask_and_repaint(image_np, updated)
            st.rerun()

    with row2_col2:
        if st.button(
            "Redo",
            use_container_width=True,
            disabled=not can_redo(),
        ):
            updated = redo_mask(mask)
            if updated is not None:
                _update_mask_and_repaint(image_np, updated)
            st.rerun()

    with row2_col3:
        if st.button(
            "Clear mask",
            use_container_width=True,
            disabled=not has_mask(mask),
        ):
            push_history(mask)
            empty = clear_editable_mask(image_np.shape)
            _update_mask_without_repaint(empty)
            app["painted_image"] = None
            app["last_painted_colour_key"] = None
            st.session_state.painted_image = None
            st.rerun()


def _get_workspace_image(image_np: np.ndarray) -> np.ndarray:
    app = get_app_state()

    mask = app.get("editable_mask")

    if app.get("show_mask", True) and has_mask(mask):
        return create_overlay(image_np, mask)

    return image_np


def _handle_workspace_click(
    image_np: np.ndarray,
    point: tuple[int, int],
    click_payload: dict,
) -> None:
    app = get_app_state()

    active_tool = app.get("active_tool", "select")
    brush_size = int(app.get("brush_size", DEFAULT_BRUSH_SIZE))

    click_signature = _build_click_signature(
        point=point,
        click_payload=click_payload,
        active_tool=active_tool,
        brush_size=brush_size,
    )

    if click_signature == app.get("last_click_signature"):
        return

    app["last_click_signature"] = click_signature

    if active_tool == "select":
        _process_surface_replace(image_np, point)

    elif active_tool == "add_surface":
        _process_surface_add(image_np, point)

    elif active_tool == "remove_surface":
        _process_surface_remove(image_np, point)

    elif active_tool in {"brush", "erase"}:
        _process_manual_mask_click(image_np, point, active_tool, brush_size)

    sync_app_to_legacy()
    st.rerun()


def _process_surface_replace(image_np: np.ndarray, seed: tuple[int, int]) -> None:
    app = get_app_state()

    current_mask = app.get("editable_mask")

    if has_mask(current_mask):
        push_history(current_mask)

    new_mask = create_wall_mask(image_np, seed)
    new_mask = set_editable_mask(new_mask)

    app["selected_surface_point"] = seed
    app["raw_mask"] = new_mask.copy()
    app["editable_mask"] = new_mask.copy()

    st.session_state.selected_surface_point = seed
    st.session_state.wall_mask = new_mask.copy()
    st.session_state.editable_mask = new_mask.copy()

    _repaint_from_mask(image_np)


def _process_surface_add(image_np: np.ndarray, seed: tuple[int, int]) -> None:
    app = get_app_state()

    current_mask = app.get("editable_mask")

    if has_mask(current_mask):
        push_history(current_mask)

    detected = create_wall_mask(image_np, seed)

    if has_mask(current_mask):
        updated = np.maximum(current_mask, detected).astype(np.uint8)
    else:
        updated = detected

    updated = set_editable_mask(updated)

    app["selected_surface_point"] = seed
    app["raw_mask"] = updated.copy()
    app["editable_mask"] = updated.copy()

    st.session_state.selected_surface_point = seed
    st.session_state.wall_mask = updated.copy()
    st.session_state.editable_mask = updated.copy()

    _repaint_from_mask(image_np)


def _process_surface_remove(image_np: np.ndarray, seed: tuple[int, int]) -> None:
    app = get_app_state()

    current_mask = app.get("editable_mask")

    if not has_mask(current_mask):
        return

    push_history(current_mask)

    detected = create_wall_mask(image_np, seed)

    updated = current_mask.copy()
    updated[detected > 0] = 0

    updated = set_editable_mask(updated)

    app["selected_surface_point"] = seed
    app["raw_mask"] = updated.copy()
    app["editable_mask"] = updated.copy()

    st.session_state.selected_surface_point = seed
    st.session_state.wall_mask = updated.copy()
    st.session_state.editable_mask = updated.copy()

    _repaint_from_mask(image_np)


def _process_manual_mask_click(
    image_np: np.ndarray,
    point: tuple[int, int],
    active_tool: str,
    brush_size: int,
) -> None:
    app = get_app_state()

    current_mask = app.get("editable_mask")

    push_history(current_mask)

    updated = apply_brush_to_mask(
        mask=current_mask,
        image_shape=image_np.shape,
        point=point,
        brush_size=brush_size,
        mode=active_tool,
    )

    _update_mask_and_repaint(image_np, updated)


def _update_mask_with_history_and_repaint(image_np: np.ndarray, mask: np.ndarray) -> None:
    app = get_app_state()

    current_mask = app.get("editable_mask")

    if has_mask(current_mask):
        push_history(current_mask)

    _update_mask_and_repaint(image_np, mask)


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
        app["last_painted_colour_key"] = None
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
    app["last_painted_colour_key"] = _colour_key(colour)

    st.session_state.painted_image = painted


def _maybe_repaint_for_colour_change(image_np: np.ndarray) -> None:
    app = get_app_state()

    mask = app.get("editable_mask")
    colour = app.get("selected_colour") or st.session_state.get("selected_colour")

    if mask is None or colour is None or not has_mask(mask):
        return

    current_colour_key = _colour_key(colour)

    if current_colour_key != app.get("last_painted_colour_key"):
        _repaint_from_mask(image_np)
        sync_app_to_legacy()


def _render_workspace_status(image_np: np.ndarray) -> None:
    app = get_app_state()

    mask = app.get("editable_mask")
    seed = app.get("selected_surface_point")
    active_tool = app.get("active_tool", "select")
    brush_size = app.get("brush_size", DEFAULT_BRUSH_SIZE)

    if seed is None:
        st.caption("Click a surface to create the first mask.")
    else:
        st.caption(f"Last click: X={seed[0]}, Y={seed[1]}")

    st.caption(f"Active tool: **{active_tool}** | Brush size: **{brush_size}px**")

    if has_mask(mask):
        st.caption(get_mask_summary(mask, image_np.shape))


def _render_output(image_np: np.ndarray) -> None:
    app = get_app_state()

    painted = app.get("painted_image")
    colour = app.get("selected_colour") or st.session_state.get("selected_colour")

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

    st.divider()
    render_export_panel(painted, colour)


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
        app["history"] = []
        app["redo_stack"] = []
        app["last_click_signature"] = None
        app["last_painted_colour_key"] = None

        st.session_state.editable_mask = None
        st.session_state.wall_mask = None
        st.session_state.painted_image = None


def _build_click_signature(
    point: tuple[int, int],
    click_payload: dict,
    active_tool: str,
    brush_size: int,
) -> str:
    app = get_app_state()

    timestamp = click_payload.get("t", "")

    return (
        f"{app.get('image_name')}_"
        f"{active_tool}_"
        f"{brush_size}_"
        f"{point[0]}_"
        f"{point[1]}_"
        f"{timestamp}"
    )


def _colour_to_rgb(colour) -> tuple[int, int, int]:
    if hasattr(colour, "to_dict"):
        colour = colour.to_dict()

    return (
        int(colour["r"]),
        int(colour["g"]),
        int(colour["b"]),
    )


def _colour_key(colour) -> str:
    if hasattr(colour, "to_dict"):
        colour = colour.to_dict()

    return (
        f"{colour.get('colour_code', '')}_"
        f"{colour.get('colour_name', '')}_"
        f"{colour.get('r', '')}_"
        f"{colour.get('g', '')}_"
        f"{colour.get('b', '')}"
    )
