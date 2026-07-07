from __future__ import annotations

import cv2
import numpy as np
import streamlit as st


def create_split_comparison(
    before_image: np.ndarray,
    after_image: np.ndarray,
    split_position: int,
) -> np.ndarray:
    """
    Create a single comparison image.

    Left side shows before.
    Right side shows after.
    """

    before = _ensure_rgb_uint8(before_image)
    after = _ensure_rgb_uint8(after_image)

    if before.shape != after.shape:
        after = cv2.resize(
            after,
            (before.shape[1], before.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    height, width = before.shape[:2]

    split_position = int(np.clip(split_position, 0, 100))
    split_x = int(width * split_position / 100)

    comparison = before.copy()
    comparison[:, split_x:] = after[:, split_x:]

    comparison = _draw_split_line(comparison, split_x)

    return comparison


def render_comparison_panel(
    before_image: np.ndarray,
    after_image: np.ndarray,
) -> None:
    """
    Render comparison controls and images.
    """

    st.subheader("Compare")

    view_mode = st.radio(
        "Comparison view",
        ["Split view", "Side by side"],
        horizontal=True,
        key="comparison_view_mode",
    )

    if view_mode == "Split view":
        split_position = st.slider(
            "Before / After split",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            key="comparison_split_position",
            help="Move the slider to reveal more or less of the painted result.",
        )

        comparison = create_split_comparison(
            before_image=before_image,
            after_image=after_image,
            split_position=split_position,
        )

        st.image(
            comparison,
            caption="Before / After split comparison",
            use_container_width=True,
        )

        st.caption(
            "Left side shows the original image. Right side shows the recoloured preview."
        )

    else:
        before_col, after_col = st.columns(2)

        with before_col:
            st.caption("Before")
            st.image(before_image, use_container_width=True)

        with after_col:
            st.caption("After")
            st.image(after_image, use_container_width=True)


def _draw_split_line(image: np.ndarray, split_x: int) -> np.ndarray:
    output = image.copy()

    height, width = output.shape[:2]

    split_x = int(np.clip(split_x, 0, width - 1))

    line_width = max(2, width // 300)

    x1 = max(0, split_x - line_width)
    x2 = min(width, split_x + line_width)

    output[:, x1:x2] = 255

    handle_radius = max(12, width // 45)
    handle_y = height // 2

    cv2.circle(
        output,
        center=(split_x, handle_y),
        radius=handle_radius,
        color=(255, 255, 255),
        thickness=-1,
    )

    cv2.circle(
        output,
        center=(split_x, handle_y),
        radius=handle_radius,
        color=(60, 60, 60),
        thickness=2,
    )

    cv2.line(
        output,
        (split_x - handle_radius // 2, handle_y),
        (split_x + handle_radius // 2, handle_y),
        (60, 60, 60),
        2,
    )

    return output


def _ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[2] == 4:
        image = image[:, :, :3]

    return image
