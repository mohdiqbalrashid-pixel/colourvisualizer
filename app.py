import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Jotun Colour Visualizer",
    page_icon="🎨",
    layout="wide"
)

# -----------------------------
# Helper functions
# -----------------------------

def resize_image(image, max_width=1000):
    width, height = image.size
    if width <= max_width:
        return image

    scale = max_width / width
    new_height = int(height * scale)
    return image.resize((max_width, new_height))


def recolor_wall_lab(image_rgb, mask, target_rgb, strength=0.85):
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    target_patch = np.uint8([[target_rgb]])
    target_lab = cv2.cvtColor(target_patch, cv2.COLOR_RGB2LAB)[0][0]

    L, A, B = cv2.split(image_lab)

    new_A = A.copy()
    new_B = B.copy()

    mask_bool = mask > 0

    new_A[mask_bool] = (
        A[mask_bool] * (1 - strength) + target_lab[1] * strength
    ).astype(np.uint8)

    new_B[mask_bool] = (
        B[mask_bool] * (1 - strength) + target_lab[2] * strength
    ).astype(np.uint8)

    recolored_lab = cv2.merge([L, new_A, new_B])
    recolored_rgb = cv2.cvtColor(recolored_lab, cv2.COLOR_LAB2RGB)

    soft_mask = cv2.GaussianBlur(mask, (25, 25), 0) / 255.0
    soft_mask = soft_mask[..., None]

    final = (
        image_rgb * (1 - soft_mask) + recolored_rgb * soft_mask
    ).astype(np.uint8)

    return final


def extract_mask_from_canvas(canvas_image):
    """
    Extract the magenta drawn area from canvas.
    The uploaded image is the background; the selection is drawn in magenta.
    """

    if canvas_image is None:
        return None

    canvas_rgb = canvas_image[:, :, :3].astype(np.uint8)

    r = canvas_rgb[:, :, 0].astype(np.int16)
    g = canvas_rgb[:, :, 1].astype(np.int16)
    b = canvas_rgb[:, :, 2].astype(np.int16)

    # Detect magenta/pink drawing overlay
    mask_bool = (
        (r > 140) &
        (b > 120) &
        (g < 120) &
        ((r - g) > 50) &
        ((b - g) > 40)
    )

    mask = mask_bool.astype(np.uint8) * 255

    # Fill and smooth the selected area
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def image_to_download_bytes(image_array):
    image = Image.fromarray(image_array)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# -----------------------------
# App
# -----------------------------

st.title("Jotun Colour Visualizer")
st.caption("Upload a photo, select the wall directly on the image, apply a colour, and export the result.")

try:
    colours = pd.read_csv("colours.csv")
except Exception as e:
    st.error("Could not load colours.csv. Please make sure it exists in your GitHub repo.")
    st.stop()

if "canvas_key" not in st.session_state:
    st.session_state["canvas_key"] = 0

uploaded_file = st.file_uploader(
    "Upload homeowner wall photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    display_image = resize_image(original_image)

    image_rgb = np.array(display_image)
    height, width = image_rgb.shape[:2]

    st.subheader("1. Select wall area directly on the image")

    st.write(
        "Draw over the wall area you want to recolour. "
        "Use **polygon** for cleaner wall edges, or **freedraw** for quick rough selection."
    )

    controls_col, canvas_col = st.columns([1, 3])

    with controls_col:
        drawing_mode = st.radio(
            "Selection tool",
            ["polygon", "freedraw"],
            index=0
        )

        stroke_width = st.slider(
            "Line thickness",
            min_value=1,
            max_value=30,
            value=3
        )

        fill_opacity = st.slider(
            "Selection visibility",
            min_value=0.10,
            max_value=0.70,
            value=0.35,
            step=0.05
        )

        if st.button("Clear selection"):
            st.session_state["canvas_key"] += 1
            st.rerun()

        st.info(
            "Tip: With polygon mode, select points around the wall. "
            "Close the shape to fill the selected area."
        )

    with canvas_col:
        canvas_result = st_canvas(
            fill_color=f"rgba(255, 0, 180, {fill_opacity})",
            stroke_width=stroke_width,
            stroke_color="#FF00B4",
            background_image=display_image,
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode=drawing_mode,
            key=f"canvas_{st.session_state['canvas_key']}"
        )

    mask = None
    if canvas_result.image_data is not None:
        mask = extract_mask_from_canvas(canvas_result.image_data)

    st.subheader("2. Choose Jotun colour")

    colour_labels = [
        f"{row['code']} — {row['name']}"
        for _, row in colours.iterrows()
    ]

    selected_label = st.selectbox("Select colour", colour_labels)

    selected_index = colour_labels.index(selected_label)
    selected_colour = colours.iloc[selected_index]

    target_rgb = (
        int(selected_colour["r"]),
        int(selected_colour["g"]),
        int(selected_colour["b"])
    )

    preview_colour = np.zeros((80, 180, 3), dtype=np.uint8)
    preview_colour[:] = target_rgb

    colour_col, details_col = st.columns([1, 3])

    with colour_col:
        st.image(preview_colour, caption="Selected colour")

    with details_col:
        st.markdown(
            f"""
            **Colour:** {selected_colour['name']}  
            **Code:** {selected_colour['code']}  
            **Product:** {selected_colour['product']}  
            **Finish:** {selected_colour['finish']}
            """
        )

    strength = st.slider(
        "Colour strength",
        min_value=0.30,
        max_value=1.00,
        value=0.85,
        step=0.05
    )

    st.subheader("3. Generate preview")

    if st.button("Apply colour", type="primary"):
        if mask is None or np.sum(mask) == 0:
            st.warning("Please select the wall area directly on the image first.")
        else:
            result = recolor_wall_lab(
                image_rgb=image_rgb,
                mask=mask,
                target_rgb=target_rgb,
                strength=strength
            )

            st.session_state["result"] = result
            st.session_state["selected_colour"] = selected_colour

    if "result" in st.session_state:
        st.subheader("4. Before / After")

        before_col, after_col = st.columns(2)

        with before_col:
            st.image(image_rgb, caption="Before", use_container_width=True)

        with after_col:
            st.image(st.session_state["result"], caption="After", use_container_width=True)

        st.subheader("5. Export")

        selected_colour = st.session_state["selected_colour"]

        st.download_button(
            label="Download recoloured image",
            data=image_to_download_bytes(st.session_state["result"]),
            file_name=f"jotun_visualizer_{selected_colour['code']}.png",
            mime="image/png"
        )

        st.caption(
            "Disclaimer: This digital preview is for visual guidance only. "
            "Actual colour appearance may vary depending on lighting, screen calibration, wall texture, surface condition, and paint finish."
        )

else:
    st.warning("Upload a room photo to start.")
