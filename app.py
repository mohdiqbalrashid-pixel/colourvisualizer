import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageOps
from io import BytesIO
from streamlit_drawable_canvas import st_canvas

# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="Jotun Colour Visualizer",
    page_icon="🎨",
    layout="wide"
)

# ============================================================
# Helper functions
# ============================================================

def resize_image(image, max_width=1000):
    width, height = image.size

    if width <= max_width:
        return image

    scale = max_width / width
    new_height = int(height * scale)

    return image.resize((max_width, new_height))


def hex_to_rgb(hex_value):
    hex_value = str(hex_value).strip().replace("#", "")

    if len(hex_value) != 6:
        return None

    try:
        return tuple(int(hex_value[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return None


def clean_colour_database(df):
    """
    Makes colours.csv flexible.
    Accepts:
    - colour_name / colour_code / hex / product
    - name / code / r / g / b / product / finish
    """

    # Clean headers
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # Rename common alternatives
    rename_map = {
        "name": "colour_name",
        "color_name": "colour_name",
        "colour": "colour_name",
        "color": "colour_name",
        "jotun_colour": "colour_name",
        "jotun_color": "colour_name",

        "code": "colour_code",
        "color_code": "colour_code",
        "colourcode": "colour_code",
        "colorcode": "colour_code",
        "jotun_code": "colour_code",

        "hex_code": "hex",
        "hex_value": "hex",
        "rgb_hex": "hex",

        "recommended_product": "product",
        "product_name": "product",
        "suggested_product": "product",
    }

    df = df.rename(columns=rename_map)

    # Required minimum fields
    required_minimum = ["colour_name", "colour_code"]

    missing_minimum = [col for col in required_minimum if col not in df.columns]

    if missing_minimum:
        st.error(
            "Your colours.csv is missing these required columns: "
            + ", ".join(missing_minimum)
        )
        st.write("Detected columns:", list(df.columns))
        st.stop()

    # If product missing, add placeholder
    if "product" not in df.columns:
        df["product"] = "Product recommendation TBC"

    # If finish missing, add placeholder
    if "finish" not in df.columns:
        df["finish"] = "Finish TBC"

    # If HEX exists, clean it
    if "hex" in df.columns:
        df["hex"] = df["hex"].astype(str).str.strip()
        df["hex"] = df["hex"].apply(lambda x: x if x.startswith("#") else f"#{x}")

    # If r/g/b missing, create from HEX
    if not all(col in df.columns for col in ["r", "g", "b"]):
        if "hex" not in df.columns:
            st.error(
                "Your colours.csv must include either HEX values or R/G/B columns."
            )
            st.write("Detected columns:", list(df.columns))
            st.stop()

        rgb_values = df["hex"].apply(hex_to_rgb)

        if rgb_values.isnull().any():
            st.error("Some HEX values in colours.csv are invalid. Use format like #C8B89F.")
            st.stop()

        df["r"] = rgb_values.apply(lambda x: x[0])
        df["g"] = rgb_values.apply(lambda x: x[1])
        df["b"] = rgb_values.apply(lambda x: x[2])

    # If HEX missing, create from r/g/b
    if "hex" not in df.columns:
        df["hex"] = df.apply(
            lambda row: "#{:02X}{:02X}{:02X}".format(
                int(row["r"]),
                int(row["g"]),
                int(row["b"])
            ),
            axis=1
        )

    # Final validation
    required_final = [
        "colour_name",
        "colour_code",
        "hex",
        "r",
        "g",
        "b",
        "product",
        "finish"
    ]

    missing_final = [col for col in required_final if col not in df.columns]

    if missing_final:
        st.error(
            "Your colours.csv is still missing: "
            + ", ".join(missing_final)
        )
        st.write("Detected columns:", list(df.columns))
        st.stop()

    # Remove empty rows
    df = df.dropna(subset=["colour_name", "colour_code", "r", "g", "b"])

    # Ensure RGB values are integers
    df["r"] = df["r"].astype(int)
    df["g"] = df["g"].astype(int)
    df["b"] = df["b"].astype(int)

    # Display label
    df["display_label"] = df.apply(
        lambda row: f"{row['colour_code']} — {row['colour_name']}",
        axis=1
    )

    return df


def recolor_wall_lab(image_rgb, mask, target_rgb, strength=0.85):
    """
    Recolour selected wall area while preserving light, shadow, and texture.
    """

    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    target_patch = np.uint8([[target_rgb]])
    target_lab = cv2.cvtColor(target_patch, cv2.COLOR_RGB2LAB)[0][0]

    L, A, B = cv2.split(image_lab)

    new_A = A.copy()
    new_B = B.copy()

    mask_bool = mask > 0

    if not np.any(mask_bool):
        return image_rgb

    new_A[mask_bool] = (
        A[mask_bool] * (1 - strength) + target_lab[1] * strength
    ).astype(np.uint8)

    new_B[mask_bool] = (
        B[mask_bool] * (1 - strength) + target_lab[2] * strength
    ).astype(np.uint8)

    recolored_lab = cv2.merge([L, new_A, new_B])
    recolored_rgb = cv2.cvtColor(recolored_lab, cv2.COLOR_LAB2RGB)

    # Feather edges for softer, more realistic transition
    soft_mask = cv2.GaussianBlur(mask, (25, 25), 0) / 255.0
    soft_mask = soft_mask[..., None]

    final = (
        image_rgb * (1 - soft_mask) + recolored_rgb * soft_mask
    ).astype(np.uint8)

    return final


def extract_mask_from_canvas(canvas_image):
    """
    Extract magenta/pink selected area from drawable canvas.
    """

    if canvas_image is None:
        return None

    canvas_rgb = canvas_image[:, :, :3].astype(np.uint8)

    r = canvas_rgb[:, :, 0].astype(np.int16)
    g = canvas_rgb[:, :, 1].astype(np.int16)
    b = canvas_rgb[:, :, 2].astype(np.int16)

    # Detect magenta/pink overlay
    mask_bool = (
        (r > 140) &
        (b > 120) &
        (g < 150) &
        ((r - g) > 35) &
        ((b - g) > 25)
    )

    mask = mask_bool.astype(np.uint8) * 255

    # Clean and fill mask
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filled_mask = np.zeros_like(mask)

    if contours:
        cv2.drawContours(
            filled_mask,
            contours,
            -1,
            255,
            thickness=cv2.FILLED
        )

    return filled_mask


def image_to_download_bytes(image_array):
    image = Image.fromarray(image_array)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def create_colour_preview(rgb_value):
    swatch = np.zeros((90, 180, 3), dtype=np.uint8)
    swatch[:] = rgb_value
    return swatch


# ============================================================
# Session state
# ============================================================

if "canvas_key" not in st.session_state:
    st.session_state["canvas_key"] = 0

if "result" not in st.session_state:
    st.session_state["result"] = None

if "selected_colour" not in st.session_state:
    st.session_state["selected_colour"] = None

if "last_uploaded_file" not in st.session_state:
    st.session_state["last_uploaded_file"] = None


# ============================================================
# App header
# ============================================================

st.title("Jotun Colour Visualizer")
st.caption(
    "Upload a room photo, select the wall directly on the image, apply a Jotun colour, and export the result."
)

st.info(
    "Prototype version: manual wall selection directly on the image. "
    "Use polygon mode for cleaner wall edges, or freedraw for quick testing."
)


# ============================================================
# Load colours
# ============================================================

try:
    colours_raw = pd.read_csv("colours.csv")
except Exception:
    st.error("Could not load colours.csv. Please make sure it exists in your GitHub repository.")
    st.stop()

colours = clean_colour_database(colours_raw)


# ============================================================
# Upload image
# ============================================================

uploaded_file = st.file_uploader(
    "Upload homeowner wall photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.warning("Upload a room photo to start.")
    st.stop()

# Open safely, including phone rotation correction
original_image = Image.open(uploaded_file)
original_image = ImageOps.exif_transpose(original_image)
original_image = original_image.convert("RGB")

display_image = resize_image(original_image)

image_rgb = np.array(display_image)
height, width = image_rgb.shape[:2]

uploaded_file_id = f"{uploaded_file.name}_{uploaded_file.size}"

if st.session_state["last_uploaded_file"] != uploaded_file_id:
    st.session_state["last_uploaded_file"] = uploaded_file_id
    st.session_state["result"] = None
    st.session_state["selected_colour"] = None
    st.session_state["canvas_key"] += 1


# ============================================================
# Wall selection
# ============================================================

st.subheader("1. Select wall area directly on the image")

st.write(
    "Draw over the wall area you want to recolour. "
    "For best results, use **polygon** and close the shape around the wall."
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
        max_value=40,
        value=3 if drawing_mode == "polygon" else 20
    )

    fill_opacity = st.slider(
        "Selection visibility",
        min_value=0.10,
        max_value=0.80,
        value=0.35,
        step=0.05
    )

    st.markdown("---")

    if st.button("Clear selection"):
        st.session_state["canvas_key"] += 1
        st.session_state["result"] = None
        st.rerun()

    st.info(
        "Tip: In polygon mode, select points around the wall and close the shape. "
        "If it feels difficult, switch to freedraw."
    )

with canvas_col:
    canvas_result = st_canvas(
        fill_color=f"rgba(255, 0, 180, {fill_opacity})",
        stroke_width=stroke_width,
        stroke_color="#FF00B4",
        background_image=display_image,
        background_color="#FFFFFF",
        update_streamlit=True,
        height=height,
        width=width,
        drawing_mode=drawing_mode,
        display_toolbar=True,
        key=f"canvas_{st.session_state['canvas_key']}_{drawing_mode}"
    )

mask = None

if canvas_result.image_data is not None:
    mask = extract_mask_from_canvas(canvas_result.image_data)


# ============================================================
# Colour selection
# ============================================================

st.subheader("2. Choose Jotun colour")

selected_label = st.selectbox(
    "Select colour",
    colours["display_label"].tolist()
)

selected_row = colours[colours["display_label"] == selected_label].iloc[0]

selected_colour_name = selected_row["colour_name"]
selected_code = selected_row["colour_code"]
selected_hex = selected_row["hex"]
selected_product = selected_row["product"]
selected_finish = selected_row["finish"]

target_rgb = (
    int(selected_row["r"]),
    int(selected_row["g"]),
    int(selected_row["b"])
)

colour_swatch = create_colour_preview(target_rgb)

colour_col, details_col = st.columns([1, 3])

with colour_col:
    st.image(colour_swatch, caption="Selected colour")

with details_col:
    st.markdown(
        f"""
        **Colour:** {selected_colour_name}  
        **Code:** {selected_code}  
        **HEX:** `{selected_hex}`  
        **Product:** {selected_product}  
        **Finish:** {selected_finish}
        """
    )

strength = st.slider(
    "Colour strength",
    min_value=0.30,
    max_value=1.00,
    value=0.85,
    step=0.05
)


# ============================================================
# Generate preview
# ============================================================

st.subheader("3. Generate preview")

generate_col, mask_col = st.columns([1, 2])

with generate_col:
    apply_clicked = st.button("Apply colour", type="primary")

with mask_col:
    if mask is not None and np.sum(mask) > 0:
        selected_pixels = int(np.sum(mask > 0))
        st.success(f"Wall area selected: {selected_pixels:,} pixels")
    else:
        st.warning("No wall area selected yet.")

if apply_clicked:
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
        st.session_state["selected_colour"] = {
            "colour_name": selected_colour_name,
            "colour_code": selected_code,
            "hex": selected_hex,
            "product": selected_product,
            "finish": selected_finish
        }


# ============================================================
# Preview and export
# ============================================================

if st.session_state["result"] is not None:
    st.subheader("4. Before / After")

    before_col, after_col = st.columns(2)

    with before_col:
        st.image(
            image_rgb,
            caption="Before",
            use_container_width=True
        )

    with after_col:
        st.image(
            st.session_state["result"],
            caption=(
                f"After — "
                f"{st.session_state['selected_colour']['colour_name']} / "
                f"{st.session_state['selected_colour']['colour_code']}"
            ),
            use_container_width=True
        )

    st.subheader("5. Export")

    export_colour = st.session_state["selected_colour"]

    st.download_button(
        label="Download recoloured image",
        data=image_to_download_bytes(st.session_state["result"]),
        file_name=f"jotun_visualizer_{export_colour['colour_code']}.png",
        mime="image/png"
    )

    st.caption(
        "Disclaimer: This digital preview is for visual guidance only. "
        "Actual colour appearance may vary depending on lighting, screen calibration, "
        "wall texture, surface condition, substrate, paint finish, and application."
    )
else:
    st.caption(
        "After selecting the wall and applying a colour, your before/after preview will appear here."
    )


# ============================================================
# Debug section
# ============================================================

with st.expander("Debug info"):
    st.write("Detected colour columns:", list(colours.columns))
    st.write("Image size:", display_image.size)
    st.write("Image mode:", display_image.mode)
    if mask is not None:
        st.write("Mask shape:", mask.shape)
        st.write("Mask selected pixels:", int(np.sum(mask > 0)))
