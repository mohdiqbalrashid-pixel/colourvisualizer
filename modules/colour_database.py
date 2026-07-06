import pandas as pd
import streamlit as st


@st.cache_data
def load_colours():

    return pd.read_csv("colours.csv")
