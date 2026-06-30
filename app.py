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

def resize_image_for_canvas(image, max_width=950):
    """
    Resize uploaded image for smoother Streamlit canvas performance.
    """
    width, height = image.size

    if width <= max_width:
        return image

    ratio = max_width / width
    new_height = int(height * ratio)
    return image.resize((max_width, new_height))


def recolor_wall_lab(image_rgb, mask, target_rgb, strength=0.85):
    """
    Recolour selected wall area while preserving light/shadow.
    image_rgb: RGB image as numpy array
    mask: binary mask, 255 where colour should apply
    target_rgb: tuple, e.g. (216, 210, 195)
    strength: colour application strength
    """

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

    # Feather edges for a more natural transition
    soft_mask = cv2.GaussianBlur(mask, (25, 25), 0) / 255.0
    soft_mask = soft_mask[..., None]

    final = (
        image_rgb * (1 - soft_mask) + recolored_rgb * soft_mask
    ).astype(np.uint8)

    return final


def extract_mask_from_canvas(canvas_image):
    """
    Extract the red drawn area from the canvas.
    The canvas includes the background image plus red polygon overlay,
    so we detect pixels that are significantly red.
    """

    if canvas_image is None:
        return None

    canvas_rgb = canvas_image[:, :, :3].astype(np.uint8)

    red = canvas_rgb[:, :, 0].astype(np.int16)
    green = canvas_rgb[:, :, 1].astype(np.int16)
    blue = canvas_rgb[:, :, 2].astype(np.int16)

    # Detect red overlay
    mask_bool = (
        (red > 120) &
        ((red - green) > 30) &
        ((red - blue) > 30)
    )

    mask = (mask_bool.astype(np.uint8)) * 255

    # Clean the mask
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
# App UI
# -----------------------------

st.title("Jotun Colour Visualizer — Manual Mask Prototype")
st.caption(
    "Upload a room photo, draw the wall area, apply a Jotun colour, and export a preview."
)

st.info(
    "Step 2 prototype: Use the drawing canvas to outline the wall/surface. "
    "This gives better control than the previous rectangle version."
)

# Load colour database
try:
    colours = pd.read_csv("colours.csv")
except Exception:
    st.error("Could not load colours.csv. Please check that the file exists in the repo.")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload homeowner wall photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    display_image = resize_image_for_canvas(original_image)

    image_rgb = np.array(display_image)
    height, width = image_rgb.shape[:2]

    st.subheader("1. Draw wall/surface area")

    st.write(
        "Draw over the wall area you want to recolour. "
        "Use the polygon tool for cleaner wall edges."
    )

    drawing_mode = st.radio(
        "Selection mode",
        ["polygon", "freedraw"],
        horizontal=True
    )

    stroke_width = st.slider(
        "Brush / outline thickness",
        min_value=1,
        max_value=30,
        value=3
    )

    canvas_result = st_canvas(
        fill_color="rgba(220, 30, 30, 0.35)",
        stroke_width=stroke_width,
        stroke_color="#DC1E1E",
        background_image=display_image,
        update_streamlit=True,
        height=height,
        width=width,
        drawing_mode=drawing_mode,
        key="wall_canvas"
    )

    mask = None

    if canvas_result.image_data is not None:
        mask = extract_mask_from_canvas(canvas_result.image_data)

    st.subheader("2. Choose Jotun colour")

    colour_labels = [
        f"{row['code']} — {row['name']}"
        for _, row in colours.iterrows()
    ]

    selected_label = st.selectbox("Select Jotun colour", colour_labels)

    selected_index = colour_labels.index(selected_label)
    selected_colour = colours.iloc[selected_index]

    target_rgb = (
        int(selected_colour["r"]),
        int(selected_colour["g"]),
        int(selected_colour["b"])
    )

    colour_preview = np.zeros((80, 180, 3), dtype=np.uint8)
    colour_preview[:] = target_rgb

    colour_col, info_col = st.columns([1, 3])

    with colour_col:
        st.image(colour_preview, caption="Selected colour", use_container_width=False)

    with info_col:
        st.markdown(
            f"""
            **Selected colour:** {selected_colour['name']}  
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
            st.warning("Please draw/select a wall area before applying colour.")
        else:
            result = recolor_wall_lab(
                image_rgb=image_rgb,
                mask=mask,
                target_rgb=target_rgb,
                strength=strength
            )

            st.session_state["result"] = result
            st.session_state["mask"] = mask
            st.session_state["selected_colour"] = selected_colour

    if "result" in st.session_state:
        result = st.session_state["result"]
        selected_colour = st.session_state["selected_colour"]

        before_col, after_col = st.columns(2)

        with before_col:
            st.image(image_rgb, caption="Before", use_container_width=True)

        with after_col:
            st.image(result, caption="After", use_container_width=True)

        st.subheader("4. Export")

        st.download_button(
            label="Download recoloured image",
            data=image_to_download_bytes(result),
            file_name=f"jotun_visualizer_{selected_colour['code']}.png",
            mime="image/png"
        )

        st.caption(
            "Disclaimer: This digital preview is for visual guidance only. "
            "Actual colour appearance may vary depending on lighting, screen calibration, wall texture, surface condition, and paint finish."
        )

else:
    st.warning("Upload a room photo to start.")
