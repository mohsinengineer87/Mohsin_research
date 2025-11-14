
# Simplified app with Asset Count + Asset Value baseline toggle
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="PASTA Model", layout="wide")

st.title("PASTA Threat Model – Asset Baseline Enhancement")

with st.sidebar:
    st.header("Asset Inputs")

    # Asset COUNT (existing variable)
    asset_count = st.slider("Asset Count (A)", 10, 10000, 300)

    # NEW: Asset VALUE baseline toggle 1–100
    use_asset_value_baseline = st.checkbox(
        "Use Asset Value Baseline (1–100)",
        value=False
    )

    asset_value = st.slider(
        "Asset Value (Scale 1–100)",
        1, 100, 50,
        help="Represents business value / impact score per asset."
    )

# Data generation example
df = pd.DataFrame({
    "AssetCount": [asset_count],
    "AssetValue": [asset_value if use_asset_value_baseline else np.random.randint(1, 101)]
})

st.subheader("Generated Values")
st.write(df)
