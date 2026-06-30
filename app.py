import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from streamlit_drawable_canvas import st_canvas


# =========================
# App setup
# =========================

st.set_page_config(
    page_title="Jotun Colour Visualizer MVP",
    layout="wide"
)

os.makedirs("outputs", exist_ok=True)


# =========================
# Helper functions
# =========================

def load_colours():
    """
    Loads colours.csv if available.
    If not available, uses a small sample palette so the app still runs.
    """
    if os.path.exists("colours.csv"):
        df = pd.read_csv("colours.csv")
    else:
        df = pd.DataFrame(
            [
                {
                    "colour_name": "Soft Sand",
                    "colour_code": "12075",
                    "hex": "#C8B89F",
                    "product": "Fenomastic Wonderwall Life",
                },
                {
                    "colour_name": "Warm White",
                    "colour_code": "1001",
                    "hex": "#F2E9DA",
                    "product": "Fenomastic Wonderwall Lux",
                },
                {
                    "colour_name": "Muted Green",
                    "colour_code": "7628",
                    "hex": "#8A927D",
                    "product": "Fenomastic Wonderwall Life",
                },
                {
                    "colour_name": "Deep Blue",
                    "colour_code": "4863",
                    "hex": "#39495C",
                    "product": "Lady Design Touch of Suede",
                },
            ]
        )

    required_cols = ["colour_name", "colour_code", "hex", "product"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing column in colours.csv: {col}")
            st.stop()

    return df


def hex_to_rgb(hex_color):
    """
    Converts HEX colour to RGB tuple.
    Example: #C8B89F -> (200, 184, 159)
    """
    hex_color = str(hex_color).strip().lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def prepare_uploaded_image(uploaded_file, max_width=900):
    """
    Opens, rotates, converts, and resizes uploaded image safely.
    """
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    w, h = image.size

    if w > max_width:
        new_h = int(h * max_width / w)
        image = image.resize((max_width, new_h))

    return image


def recolour_wall(image_pil, mask, target_hex, strength=0.85, lightness_strength=0.35):
    """
    Recolours selected wall area while preserving lighting and texture.

    image_pil: PIL RGB image
    mask: binary mask, wall = 255, background = 0
    target_hex: target colour in HEX
    strength: how strongly colour is applied
    lightness_strength: how much target lightness influences original wall
    """

    image_rgb = np.array(image_pil.convert("RGB"))

    # Ensure mask matches image dimensions
    if mask.shape[:2] != image_rgb.shape[:2]:
        mask = cv2.resize(
            mask,
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    mask_binary = mask > 0

    if not np.any(mask_binary):
        return image_pil

    # Convert image to LAB
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(image_lab)

    # Convert target colour to LAB
    target_rgb = np.uint8([[hex_to_rgb(target_hex)]])
    target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB)[0][0]

    original_L = L.copy()
    original_A = A.copy()
    original_B = B.copy()

    # Blend A and B colour channels
    A[mask_binary] = (
        original_A[mask_binary] * (1 - strength) + target_lab[1] * strength
    ).astype(np.uint8)

    B[mask_binary] = (
        original_B[mask_binary] * (1 - strength) + target_lab[2] * strength
    ).astype(np.uint8)

    # Adjust lightness gently while preserving shadows
    wall_L_median = np.median(original_L[mask_binary])
    target_L = target_lab[0]
    adjustment = (target_L - wall_L_median) * lightness_strength

    L[mask_binary] = np.clip(
        original_L[mask_binary].astype(np.float32) + adjustment,
        0,
        255
    ).astype(np.uint8)

    recoloured_lab = cv2.merge([L, A, B])
    recoloured_rgb = cv2.cvtColor(recoloured_lab, cv2.COLOR_LAB2RGB)

    # Feather mask edges for cleaner blending
    soft_mask = cv2.GaussianBlur(mask.astype(np.float32) / 255, (21, 21), 0)
    soft_mask = np.expand_dims(soft_mask, axis=-1)

    final_rgb = (
        recoloured_rgb * soft_mask + image_rgb * (1 - soft_mask)
    ).astype(np.uint8)

    return Image.fromarray(final_rgb)


def mask_from_canvas_json(json_data, image_size):
    """
    Creates a binary mask from drawable canvas JSON objects.
    Works best for rectangle mode.
    """
    width, height = image_size
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    if not json_data or "objects" not in json_data:
        return np.array(mask_img)

    objects = json_data.get("objects", [])

    for obj in objects:
        obj_type = obj.get("type", "")

        if obj_type == "rect":
            left = int(obj.get("left", 0))
            top = int(obj.get("top", 0))
            obj_width = int(obj.get("width", 0) * obj.get("scaleX", 1))
            obj_height = int(obj.get("height", 0) * obj.get("scaleY", 1))

            right = left + obj_width
            bottom = top + obj_height

            draw.rectangle([left, top, right, bottom], fill=255)

        elif obj_type == "circle":
            left = int(obj.get("left", 0))
            top = int(obj.get("top", 0))
            radius = int(obj.get("radius", 0) * obj.get("scaleX", 1))

            draw.ellipse(
                [left, top, left + radius * 2, top + radius * 2],
                fill=255
            )

    return np.array(mask_img)


def mask_from_canvas_pixels(canvas_image_data, image_size):
    """
    Fallback mask extraction from red drawn pixels.
    Useful for freedraw mode.
    """
    width, height = image_size

    if canvas_image_data is None:
        return np.zeros((height, width), dtype=np.uint8)

    data = np.array(canvas_image_data).astype(np.uint8)

    # If canvas has RGBA, use first 3 channels
    rgb = data[:, :, :3]

    # Detect red drawing overlay
    red_mask = (
        (rgb[:, :, 0] > 150) &
        (rgb[:, :, 1] < 120) &
        (rgb[:, :, 2] < 120)
    )

    mask = np.zeros(red_mask.shape, dtype=np.uint8)
    mask[red_mask] = 255

    # Smooth and slightly expand mask
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    return mask


def create_rectangle_mask(image_size, x1, y1, x2, y2):
    """
    Creates rectangle mask from slider coordinates.
    """
    width, height = image_size
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.rectangle([x1, y1, x2, y2], fill=255)
    return np.array(mask_img)


def create_mask_preview(image_pil, mask):
    """
    Creates red transparent overlay preview of selected wall mask.
    """
    image_rgba = image_pil.convert("RGBA")
    overlay = Image.new("RGBA", image_pil.size, (0, 0, 0, 0))

    mask_img = Image.fromarray(mask).convert("L")
    red_layer = Image.new("RGBA", image_pil.size, (220, 30, 30, 100))

    overlay.paste(red_layer, (0, 0), mask_img)
    preview = Image.alpha_composite(image_rgba, overlay)

    return preview


# =========================
# Sidebar
# =========================

st.sidebar.title("Settings")

max_width = st.sidebar.slider(
    "Canvas image width",
    min_value=500,
    max_value=1200,
    value=900,
    step=50
)

selection_method = st.sidebar.radio(
    "Wall selection method",
    [
        "Draw on image",
        "Fallback rectangle sliders"
    ]
)

drawing_mode = st.sidebar.selectbox(
    "Drawing mode",
    [
        "rect",
        "freedraw"
    ],
    index=0
)

stroke_width = st.sidebar.slider(
    "Brush / border thickness",
    min_value=3,
    max_value=80,
    value=30 if drawing_mode == "freedraw" else 3
)

colour_strength = st.sidebar.slider(
    "Colour strength",
    min_value=0.10,
    max_value=1.00,
    value=0.85,
    step=0.05
)

lightness_strength = st.sidebar.slider(
    "Lightness adjustment",
    min_value=0.00,
    max_value=0.75,
    value=0.35,
    step=0.05
)


# =========================
# Main app
# =========================

st.title("Jotun Colour Visualizer MVP")
st.caption("Consultant-facing prototype: upload a wall photo, select the surface, apply colour, and export a preview.")

colours = load_colours()

uploaded_file = st.file_uploader(
    "Upload homeowner wall photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Upload a room or wall photo to begin.")
    st.stop()


# =========================
# Load image
# =========================

image_key = f"{uploaded_file.name}_{uploaded_file.size}_{max_width}"

if st.session_state.get("image_key") != image_key:
    image = prepare_uploaded_image(uploaded_file, max_width=max_width)

    st.session_state["image_key"] = image_key
    st.session_state["uploaded_image"] = image
    st.session_state["mask"] = None
else:
    image = st.session_state["uploaded_image"]

width, height = image.size


# =========================
# Show original
# =========================

st.subheader("1. Uploaded image")

st.image(
    image,
    caption=f"Image loaded successfully — size: {width} x {height}, mode: {image.mode}",
    use_container_width=True
)


# =========================
# Wall selection
# =========================

st.subheader("2. Select the wall area")

mask = None

if selection_method == "Draw on image":
    st.info(
        "If the canvas shows a black screen, switch the sidebar option to "
        "'Fallback rectangle sliders'."
    )

    canvas_result = st_canvas(
        fill_color="rgba(220, 30, 30, 0.35)",
        stroke_width=stroke_width,
        stroke_color="#DC1E1E",
        background_color="#FFFFFF",
        background_image=image,
        update_streamlit=True,
        height=height,
        width=width,
        drawing_mode=drawing_mode,
        display_toolbar=True,
        key=f"wall_canvas_{image_key}_{drawing_mode}_{stroke_width}"
    )

    # First try JSON extraction for rectangles/circles
    if canvas_result.json_data is not None:
        mask_json = mask_from_canvas_json(canvas_result.json_data, image.size)
    else:
        mask_json = np.zeros((height, width), dtype=np.uint8)

    # Then try red pixel extraction for freehand drawing
    mask_pixels = mask_from_canvas_pixels(canvas_result.image_data, image.size)

    # Combine both
    mask = np.maximum(mask_json, mask_pixels)

else:
    st.write("Use the sliders below to create a simple rectangular wall mask.")

    col_a, col_b = st.columns(2)

    with col_a:
        x1 = st.slider("Left", 0, width, int(width * 0.15))
        x2 = st.slider("Right", 0, width, int(width * 0.85))

    with col_b:
        y1 = st.slider("Top", 0, height, int(height * 0.15))
        y2 = st.slider("Bottom", 0, height, int(height * 0.85))

    if x2 <= x1:
        st.warning("Right must be greater than Left.")
        st.stop()

    if y2 <= y1:
        st.warning("Bottom must be greater than Top.")
        st.stop()

    mask = create_rectangle_mask(image.size, x1, y1, x2, y2)


# Save mask
if mask is not None:
    st.session_state["mask"] = mask


# =========================
# Mask preview
# =========================

st.subheader("3. Mask preview")

mask = st.session_state.get("mask")

if mask is None or np.sum(mask) == 0:
    st.warning("No wall area selected yet. Draw a rectangle over the wall or use the fallback sliders.")
    st.stop()

mask_preview = create_mask_preview(image, mask)

st.image(
    mask_preview,
    caption="Selected wall area preview",
    use_container_width=True
)


# =========================
# Colour selection
# =========================

st.subheader("4. Select Jotun colour")

selected_colour = st.selectbox(
    "Choose colour",
    colours["colour_name"].tolist()
)

selected_row = colours[colours["colour_name"] == selected_colour].iloc[0]

selected_hex = selected_row["hex"]
selected_code = selected_row["colour_code"]
selected_product = selected_row["product"]

colour_chip_col, colour_info_col = st.columns([1, 4])

with colour_chip_col:
    st.markdown(
        f"""
        <div style="
            width: 100%;
            height: 90px;
            border-radius: 14px;
            border: 1px solid #ddd;
            background: {selected_hex};
        "></div>
        """,
        unsafe_allow_html=True
    )

with colour_info_col:
    st.markdown(
        f"""
        **Selected colour:** {selected_colour}  
        **Colour code:** {selected_code}  
        **Digital HEX:** `{selected_hex}`  
        **Suggested product:** {selected_product}
        """
    )


# =========================
# Apply colour
# =========================

st.subheader("5. Apply colour")

if st.button("Apply selected colour", type="primary"):
    result = recolour_wall(
        image_pil=image,
        mask=mask,
        target_hex=selected_hex,
        strength=colour_strength,
        lightness_strength=lightness_strength
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"jotun_colour_visualizer_{timestamp}.png"
    output_path = os.path.join("outputs", output_filename)

    result.save(output_path)

    st.success("Colour applied successfully.")

    before_col, after_col = st.columns(2)

    with before_col:
        st.image(
            image,
            caption="Before",
            use_container_width=True
        )

    with after_col:
        st.image(
            result,
            caption=f"After — {selected_colour} / {selected_code}",
            use_container_width=True
        )

    with open(output_path, "rb") as file:
        st.download_button(
            label="Download recoloured image",
            data=file,
            file_name=output_filename,
            mime="image/png"
        )

    st.warning(
        "This is a digital visual preview only. Final appearance may vary depending on lighting, "
        "surface texture, screen calibration, substrate, finish, and paint application."
    )


# =========================
# Debug info
# =========================

with st.expander("Debug info"):
    st.write("Image size:", image.size)
    st.write("Image mode:", image.mode)
    st.write("Mask shape:", mask.shape if mask is not None else None)
    st.write("Mask selected pixels:", int(np.sum(mask > 0)) if mask is not None else 0)
