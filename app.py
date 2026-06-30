import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title="Jotun Colour Visualizer",
    page_icon="🎨",
    layout="wide"
)

# -----------------------------
# Helper functions
# -----------------------------

def hex_to_rgb(hex_code):
    hex_code = hex_code.strip().replace("#", "")
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))


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

    soft_mask = cv2.GaussianBlur(mask, (25, 25), 0) / 255.0
    soft_mask = soft_mask[..., None]

    final = (
        image_rgb * (1 - soft_mask) + recolored_rgb * soft_mask
    ).astype(np.uint8)

    return final


def image_to_download_bytes(image_array):
    image = Image.fromarray(image_array)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# -----------------------------
# App UI
# -----------------------------

st.title("Jotun Colour Visualizer — Prototype")
st.caption(
    "Upload a room photo, select a wall area, apply a Jotun colour, and export a preview."
)

st.info(
    "Prototype note: This first version uses a rectangular wall selection. "
    "The next version can add brush selection and AI-assisted wall detection."
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
    image_rgb = np.array(original_image)

    height, width = image_rgb.shape[:2]

    st.subheader("1. Select wall area")

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.image(image_rgb, caption="Original image", use_container_width=True)

    with right_col:
        st.write("Adjust the box to cover the wall area you want to recolour.")

        x1 = st.slider("Left edge", 0, width - 1, int(width * 0.15))
        x2 = st.slider("Right edge", 1, width, int(width * 0.85))
        y1 = st.slider("Top edge", 0, height - 1, int(height * 0.15))
        y2 = st.slider("Bottom edge", 1, height, int(height * 0.85))

        if x2 <= x1 or y2 <= y1:
            st.warning("Please make sure right edge is after left edge, and bottom edge is below top edge.")
            st.stop()

        preview = image_rgb.copy()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 0), 4)
        st.image(preview, caption="Selected wall area", use_container_width=True)

    st.subheader("2. Choose colour")

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

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    result = recolor_wall_lab(
        image_rgb=image_rgb,
        mask=mask,
        target_rgb=target_rgb,
        strength=strength
    )

    st.subheader("3. Preview result")

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
