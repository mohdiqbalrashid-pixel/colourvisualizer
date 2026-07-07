from __future__ import annotations

import pandas as pd
import streamlit as st


def _normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    rename_map = {
        "code": "colour_code",
        "color_code": "colour_code",
        "colourcode": "colour_code",
        "colorcode": "colour_code",
        "jotun_code": "colour_code",

        "name": "colour_name",
        "color_name": "colour_name",
        "colour": "colour_name",
        "color": "colour_name",
        "jotun_colour": "colour_name",
        "jotun_color": "colour_name",

        "hex_code": "hex",
        "hex_value": "hex",
        "rgb_hex": "hex",

        "recommended_product": "product",
        "product_name": "product",
        "suggested_product": "product",
    }

    return df.rename(columns=rename_map)


def _hex_to_rgb(hex_value: str) -> tuple[int, int, int] | None:
    value = str(hex_value).strip().replace("#", "")

    if len(value) != 6:
        return None

    try:
        return (
            int(value[0:2], 16),
            int(value[2:4], 16),
            int(value[4:6], 16),
        )
    except ValueError:
        return None


def _clean_hex(value: str) -> str:
    value = str(value).strip()

    if not value.startswith("#"):
        value = f"#{value}"

    return value.upper()


@st.cache_data(show_spinner=False)
def load_colours() -> pd.DataFrame:
    df = pd.read_csv("colours.csv")

    df = _normalise_column_names(df)

    required_basic_columns = ["colour_code", "colour_name"]
    missing_basic_columns = [
        column for column in required_basic_columns if column not in df.columns
    ]

    if missing_basic_columns:
        st.error(
            "Your colours.csv is missing required columns: "
            + ", ".join(missing_basic_columns)
        )
        st.stop()

    if "product" not in df.columns:
        df["product"] = "Product recommendation TBC"

    if "finish" not in df.columns:
        df["finish"] = "Finish TBC"

    if "lrv" not in df.columns:
        df["lrv"] = ""

    has_rgb = all(column in df.columns for column in ["r", "g", "b"])
    has_hex = "hex" in df.columns

    if not has_rgb and not has_hex:
        st.error("Your colours.csv must include either HEX or R/G/B values.")
        st.stop()

    if has_hex:
        df["hex"] = df["hex"].apply(_clean_hex)

    if not has_rgb:
        rgb_values = df["hex"].apply(_hex_to_rgb)

        if rgb_values.isnull().any():
            st.error("Some HEX values in colours.csv are invalid.")
            st.stop()

        df["r"] = rgb_values.apply(lambda value: value[0])
        df["g"] = rgb_values.apply(lambda value: value[1])
        df["b"] = rgb_values.apply(lambda value: value[2])

    df["r"] = pd.to_numeric(df["r"], errors="coerce")
    df["g"] = pd.to_numeric(df["g"], errors="coerce")
    df["b"] = pd.to_numeric(df["b"], errors="coerce")

    df = df.dropna(subset=["colour_code", "colour_name", "r", "g", "b"])

    df["r"] = df["r"].astype(int).clip(0, 255)
    df["g"] = df["g"].astype(int).clip(0, 255)
    df["b"] = df["b"].astype(int).clip(0, 255)

    if "hex" not in df.columns:
        df["hex"] = df.apply(
            lambda row: "#{:02X}{:02X}{:02X}".format(
                int(row["r"]),
                int(row["g"]),
                int(row["b"]),
            ),
            axis=1,
        )

    df["colour_code"] = df["colour_code"].astype(str)
    df["colour_name"] = df["colour_name"].astype(str)
    df["product"] = df["product"].astype(str)
    df["finish"] = df["finish"].astype(str)

    df["display_label"] = (
        df["colour_code"]
        + " • "
        + df["colour_name"]
    )

    return df.reset_index(drop=True)
