import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageOps
from io import BytesIO
from streamlit_image_coordinates import streamlit_image_coordinates

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
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

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

    required_minimum = ["colour_name", "colour_code"]
    missing_minimum = [col for col in required_minimum if col not in df.columns]

    if missing_minimum:
        st.error(
            "Your colours.csv is missing these required columns: "
            + ", ".join(missing_minimum)
        )
        st.write("Detected columns:", list(df.columns))
        st.stop()

    if "product" not in df.columns:
        df["product"] = "Product recommendation TBC"

    if "finish" not in df.columns:
        df["finish"] = "Finish TBC"

    if "hex" in df.columns:
        df["hex"] = df["hex"].astype(str).str.strip()
        df["hex"] = df["hex"].apply(lambda x: x if x.startswith("#") else f"#{x}")

    if not all(col in df.columns for col in ["r", "g", "b"]):
        if "hex" not in df.columns:
            st.error("Your colours.csv must include either HEX values or R/G/B columns.")
            st.write("Detected columns:", list(df.columns))
            st.stop()

        rgb_values = df["hex"].apply(hex_to_rgb)

        if rgb_values.isnull().any():
            st.error("Some HEX values in colours.csv are invalid. Use format like #C8B89F.")
            st.stop()

        df["r"] = rgb_values.apply(lambda x: x[0])
        df["g"] = rgb_values.apply(lambda x: x[1])
        df["b"] = rgb_values.apply(lambda x: x[2])

    if "hex" not in df.columns:
        df["hex"] = df.apply(
            lambda row: "#{:02X}{:02X}{:02X}".format(
                int(row["r"]),
                int(row["g"]),
                int(row["b"])
            ),
            axis=1
        )

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
        st.error("Your colours.csv is still missing: " + ", ".join(missing_final))
        st.write("Detected columns:", list(df.columns))
        st.stop()

    df = df.dropna(subset=["colour_name", "colour_code", "r", "g", "b"])

    df["r"] = df["r"].astype(int)
    df["g"] = df["g"].astype(int)
    df["b"] = df["b"].astype(int)

    df["display_label"] = df.apply(
        lambda row: f"{row['colour_code']} — {row['colour_name']}",
        axis=1
    )

    return df


def draw_points_on_image(image, points, closed=False):
    preview = image.copy()
    draw = ImageDraw.Draw(preview, "RGBA")

    if len(points) >= 2:
        line_points = points.copy()

        if closed and len(points) >= 3:
            line_points = points + [points[0]]

        draw.line(line_points, fill=(255, 0, 180, 255), width=4)

    for i, point in enumerate(points):
        x, y = point
        radius = 7

        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(255, 0, 180, 255),
            outline=(255, 255, 255, 255),
            width=2
        )

        draw.text(
            (x + 10, y - 10),
            str(i + 1),
            fill=(255, 255, 255, 255)
        )

    if closed and len(points) >= 3:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay, "RGBA")
        overlay_draw.polygon(points, fill=(255, 0, 180, 70))
        preview = Image.alpha_composite(preview.convert("RGBA"), overlay).convert("RGB")

    return preview


def create_polygon_mask(image_size, points):
    width, height = image_size
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    if len(points) >= 3:
        draw.polygon(points, fill=255)

    return np.array(mask_img)


def recolor_wall_lab(image_rgb, mask, target_rgb, strength=0.85):
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


def create_colour_preview(rgb_value):
    swatch = np.zeros((90, 180, 3), dtype=np.uint8)
    swatch[:] = rgb_value
    return swatch


# ============================================================
# Session state
# ============================================================

if "points" not in st.session_state:
    st.session_state["points"] = []

if "polygon_closed" not in st.session_state:
    st.session_state["polygon_closed"] = False

if "result" not in st.session_state:
    st.session_state["result"] = None

if "selected_colour" not in st.session_state:
    st.session_state["selected_colour"] = None

if "last_uploaded_file" not in st.session_state:
    st.session_state["last_uploaded_file"] = None

if "last_click_signature" not in st.session_state:
    st.session_state["last_click_signature"] = None


# ============================================================
# App header
# ============================================================

st.title("Jotun Colour Visualizer")
st.caption(
    "Upload a room photo, select wall corners directly on the image, apply a Jotun colour, and export the result."
)

st.info(
    "This version avoids the black canvas issue by using direct image click coordinates instead of drawable canvas background images."
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

original_image = Image.open(uploaded_file)
original_image = ImageOps.exif_transpose(original_image)
original_image = original_image.convert("RGB")

display_image = resize_image(original_image)

image_rgb = np.array(display_image)
height, width = image_rgb.shape[:2]

uploaded_file_id = f"{uploaded_file.name}_{uploaded_file.size}"

if st.session_state["last_uploaded_file"] != uploaded_file_id:
    st.session_state["last_uploaded_file"] = uploaded_file_id
    st.session_state["points"] = []
    st.session_state["polygon_closed"] = False
    st.session_state["result"] = None
    st.session_state["selected_colour"] = None
    st.session_state["last_click_signature"] = None


# ============================================================
# Wall selection
# ============================================================

st.subheader("1. Select wall area directly on the uploaded image")

st.write(
    "Click around the wall corners in order. Use at least **3 points**, then select **Close wall shape**."
)

control_col, image_col = st.columns([1, 3])

with control_col:
    st.markdown("### Selection controls")

    st.write(f"Points selected: **{len(st.session_state['points'])}**")

    if st.button("Undo last point"):
        if st.session_state["points"]:
            st.session_state["points"].pop()
            st.session_state["polygon_closed"] = False
            st.session_state["result"] = None
            st.rerun()

    if st.button("Clear all points"):
        st.session_state["points"] = []
        st.session_state["polygon_closed"] = False
        st.session_state["result"] = None
        st.session_state["last_click_signature"] = None
        st.rerun()

    if st.button("Close wall shape"):
        if len(st.session_state["points"]) >= 3:
            st.session_state["polygon_closed"] = True
            st.session_state["result"] = None
            st.rerun()
        else:
            st.warning("Select at least 3 points before closing the wall shape.")

    st.info(
        "Tip: Click the main corners of the wall. You do not need too many points at first."
    )

with image_col:
    preview_image = draw_points_on_image(
        display_image,
        st.session_state["points"],
        closed=st.session_state["polygon_closed"]
    )

    click_value = streamlit_image_coordinates(
        preview_image,
        key=f"image_clicker_{uploaded_file_id}_{len(st.session_state['points'])}_{st.session_state['polygon_closed']}",
        use_column_width=True,
        cursor="crosshair"
    )

    if click_value is not None and not st.session_state["polygon_closed"]:
        x = int(click_value["x"])
        y = int(click_value["y"])
        click_time = click_value.get("t", None)

        click_signature = f"{x}_{y}_{click_time}"

        if st.session_state["last_click_signature"] != click_signature:
            st.session_state["points"].append((x, y))
            st.session_state["last_click_signature"] = click_signature
            st.rerun()


mask = None

if st.session_state["polygon_closed"] and len(st.session_state["points"]) >= 3:
    mask = create_polygon_mask(display_image.size, st.session_state["points"])

    selected_pixels = int(np.sum(mask > 0))
    st.success(f"Wall shape closed. Selected area: {selected_pixels:,} pixels.")
else:
    st.warning("Wall shape is not closed yet. Select points and then choose 'Close wall shape'.")


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

if st.button("Apply colour", type="primary"):
    if mask is None or np.sum(mask) == 0:
        st.warning("Please select and close the wall shape first.")
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
# Debug info
# ============================================================

with st.expander("Debug info"):
    st.write("Image size:", display_image.size)
    st.write("Image mode:", display_image.mode)
    st.write("Points:", st.session_state["points"])
    st.write("Polygon closed:", st.session_state["polygon_closed"])
    st.write("Detected colour columns:", list(colours.columns))
