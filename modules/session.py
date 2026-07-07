from __future__ import annotations

from typing import Any

import streamlit as st


def _default_app_state() -> dict[str, Any]:
    return {
        "image": None,
        "image_name": None,
        "selected_colour": None,
        "selected_surface_point": None,
        "raw_mask": None,
        "editable_mask": None,
        "painted_image": None,
        "show_mask": True,
        "history": [],
        "redo_stack": [],
        "messages": [],
    }


def initialise_session() -> None:
    """
    Initialise both the new v2 application state and the older direct
    Streamlit keys.

    The older keys are kept for now so the existing sidebar and preview
    modules continue to work while we migrate module by module.
    """

    if "app" not in st.session_state:
        st.session_state.app = _default_app_state()

    legacy_defaults = {
        "uploaded_image": None,
        "selected_colour": None,
        "selected_surface_point": None,
        "wall_mask": None,
        "painted_image": None,
        "editable_mask": None,
        "show_mask": True,
    }

    for key, value in legacy_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_app_state() -> dict[str, Any]:
    if "app" not in st.session_state:
        initialise_session()

    return st.session_state.app


def sync_legacy_to_app() -> None:
    """
    Temporary bridge while older modules still use direct session keys.
    We will remove this once all modules are migrated to st.session_state.app.
    """

    app = get_app_state()

    app["image"] = st.session_state.get("uploaded_image")
    app["selected_colour"] = st.session_state.get("selected_colour")
    app["selected_surface_point"] = st.session_state.get("selected_surface_point")
    app["raw_mask"] = st.session_state.get("wall_mask")
    app["editable_mask"] = st.session_state.get("editable_mask")
    app["painted_image"] = st.session_state.get("painted_image")
    app["show_mask"] = st.session_state.get("show_mask", True)


def sync_app_to_legacy() -> None:
    """
    Temporary bridge so the current modules can keep working.
    """

    app = get_app_state()

    st.session_state.uploaded_image = app.get("image")
    st.session_state.selected_colour = app.get("selected_colour")
    st.session_state.selected_surface_point = app.get("selected_surface_point")
    st.session_state.wall_mask = app.get("raw_mask")
    st.session_state.editable_mask = app.get("editable_mask")
    st.session_state.painted_image = app.get("painted_image")
    st.session_state.show_mask = app.get("show_mask", True)


def reset_project() -> None:
    """
    Clear all image-specific state while keeping the app loaded.
    """

    app = get_app_state()

    app["image"] = None
    app["image_name"] = None
    app["selected_surface_point"] = None
    app["raw_mask"] = None
    app["editable_mask"] = None
    app["painted_image"] = None
    app["history"] = []
    app["redo_stack"] = []
    app["messages"] = []

    sync_app_to_legacy()


def reset_masks() -> None:
    """
    Clear mask and painted output but keep image and selected colour.
    """

    app = get_app_state()

    app["selected_surface_point"] = None
    app["raw_mask"] = None
    app["editable_mask"] = None
    app["painted_image"] = None
    app["history"] = []
    app["redo_stack"] = []

    sync_app_to_legacy()
