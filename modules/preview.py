click = streamlit_image_coordinates(
    image_np,
    key="image_click"
)

if click:

    seed = (click["x"], click["y"])

    # Prevent repeated recomputation for same click
    if seed != st.session_state.selected_surface_point:

        st.session_state.selected_surface_point = seed

        mask = create_wall_mask(image_np, seed)

        st.session_state.wall_mask = mask

        # Apply paint only if colour selected
        if st.session_state.selected_colour is not None:

            from modules.recolouring import apply_paint

            painted = apply_paint(
                image_np,
                mask,
                (
                    st.session_state.selected_colour["r"],
                    st.session_state.selected_colour["g"],
                    st.session_state.selected_colour["b"]
                )
            )

            st.session_state.painted_image = painted

        st.success(f"Wall refined from {seed}")
