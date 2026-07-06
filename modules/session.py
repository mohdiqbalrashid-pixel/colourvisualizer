import streamlit as st


DEFAULT_STATE = {

    "uploaded_image": None,

    "selected_colour": None,

    "selected_surface_point": None,

    "wall_mask": None,

    "painted_image": None

}


def initialise_session():

    for key, value in DEFAULT_STATE.items():

        if key not in st.session_state:

            st.session_state[key] = value
