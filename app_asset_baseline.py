import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from sklearn.metrics import r2_score

# ---------- Industry Baseline (Assets only for now) ----------
INDUSTRY_BASELINES = {
    "Assets": 5000   # NIST/ISO 27005 baseline example
}

# ---------- Page ----------
st.set_page_config(page_title="PASTA Effort Estimator", layout="wide")
st.title("📊 PASTA Effort Estimator — Noise, Log Mode & Sensitivity")

# ---------- Sidebar: Controls ----------
with st.sidebar:
    st.header("🔧 Dataset Settings")
    num_samples = st.slider("Number of Samples", 50, 20000, 200, help="How many rows to generate.")
    scaling_factor = st.slider("Scaling Factor (S)", 0.1, 3.0, 1.1, 0.1)
    rng_seed = st.number_input("Random Seed", value=42, help="For reproducibility.")

    st.header("🎛️ Attribute Ranges")

    # ------------------ ASSETS WITH BASELINE TOGGLE ------------------
    asset_range = st.slider("Assets (A)", 10, 10000, (100, 300))

    use_asset_baseline = st.checkbox(
        "Use NIST/ISO 27005 Baseline for Assets",
        value=False,
        help="When enabled, all samples use a fixed NIST/ISO baseline asset count."
    )
    # -----------------------------------------------------------------

    threat_range = st.slider("Threat Vectors (T)", 1, 5000, (10, 80))
    vuln_range = st.slider("Vulnerabilities (V)", 1, 20000, (30, 200))
    complexity_range = st.slider("Complexity (C)", 1, 10, (1, 5))
    change_rate_range = st.slider("Change Rate (R)", 1, 10, (1, 5))
    maturity_range = st.slider("Org Maturity (M)", 1, 10, (1, 5))
    automation_range = st.slider("Automation (Au)", 1, 10, (1, 5))

    st.header("🔊 Noise Settings")
    noise_type = st.selectbox("Noise Distribution", ["None", "Gaussian", "Laplace"])
    noise_pct = st.slider(
        "Noise Scale (% of mean Predicted)", 0.0, 100.0, 5.0, 0.5,
        help="Standard deviation (Gaussian) or b (Laplace) as % of mean(Predicted)."
    )
    noise_seed = st.number_input("Noise Seed", value=123, help="Separate seed for noise.")

    st.header("📐 Analysis Settings")
    log_mode = st.checkbox(
        "Analyze/Plot in log10 space", value=False,
        help="R² and plots computed on log10(Actual) vs log10(Predicted)."
    )
    ofat_points = st.slider("OFAT sweep points", 10, 200, 50,
                            help="Number of points in one-factor-at-a-time sweep.")
    selected_var = st.selectbox(
        "Sensitivity: variable to sweep (OFAT)",
        ["Assets", "ThreatVectors", "Vulnerabilities", "Complexity", "ChangeRate", "OrgMaturity", "Automation"]
    )

# ---------- Helpers ----------
def generate_data(n, ranges, seed, use_asset_baseline=False, baselines=None):
    np.random.seed(seed)
    baselines = baselines or {}

    # ASSETS with baseline override
    if use_asset_baseline:
        A = np.full(n, baselines["Assets"], dtype=int)
    else:
        A = np.random.randint(ranges["Assets"][0], ranges["Assets"][1] + 1, n)

    # All other variables unchanged
    T = np.random.randint(ranges["ThreatVectors"][0], ranges["ThreatVectors"][1] + 1, n)
    V = np.random.randint(ranges["Vulnerabilities"][0], ranges["Vulnerabilities"][1] + 1, n)
    C = np.random.randint(ranges["Complexity"][0], ranges["Complexity"][1] + 1, n)
    R = np.random.randint(ranges["ChangeRate"][0], ranges["ChangeRate"][1] + 1, n)
    M = np.random.randint(ranges["OrgMaturity"][0], ranges["OrgMaturity"][1] + 1, n)
    Au = np.random.randint(ranges["Automation"][0], ranges["Automation"][1] + 1, n)

    return pd.DataFrame({
        "Assets": A, "ThreatVectors": T, "Vulnerabilities": V,
        "Complexity": C, "ChangeRate": R, "OrgMaturity": M, "Automation": Au
    })

def model_predicted(df, S):
    num = df["Assets"] * df["ThreatVectors"] * df["Vulnerabilities"] * df["Complexity"] * df["ChangeRate"]
    den = df["OrgMaturity"] * df["Automation"]
    den = den.replace(0, 1)  # defensive
    return (num.astype(np.float64)) ** S / den

def add_noise(y_pred, noise_type, noise_pct, seed):
    if noise_type == "None" or noise_pct == 0.0:
        return y_pred.copy()
    rng = np.random.default_rng(seed)
    scale = (noise_pct / 100.0) * float(np.mean(y_pred))
    if noise_type == "Gaussian":
        noise = rng.normal(loc=0.0, scale=scale, size=len(y_pred))
    else:  # Laplace
        noise = rng.laplace(loc=0.0, scale=scale, size=len(y_pred))
    return y_pred + noise

def safe_log10(x):
    x = np.asarray(x, dtype=np.float64)
    min_pos = np.min(x[x > 0]) if np.any(x > 0) else 1.0
    eps = min_pos * 1e-9
    return np.log10(np.clip(x, eps, None))

def figure_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    buf.seek(0)
    return buf

# ---------- Generate data & targets ----------
ranges = {
    "Assets": asset_range, "ThreatVectors": threat_range, "Vulnerabilities": vuln_range,
    "Complexity": complexity_range, "ChangeRate": change_rate_range,
    "OrgMaturity": maturity_range, "Automation": automation_range,
}

df = generate_data(
    num_samples, ranges, seed=rng_seed,
    use_asset_baseline=use_asset_baseline,
    baselines=INDUSTRY_BASELINES
)
df["PredictedEffort"] = model_predicted(df, scaling_factor)
df["ActualEffort"] = add_noise(df["PredictedEffort"], noise_type, noise_pct, seed=noise_seed)

# ---------- Metrics ----------
if log_mode:
    y_true = safe_log10(df["ActualEffort"])
    y_pred = safe_log10(df["PredictedEffort"])
    r2_log = r2_score(y_true, y_pred)
    r2_raw = r2_score(df["ActualEffort"], df["PredictedEffort"])
else:
    y_true = df["ActualEffort"].values
    y_pred = df["PredictedEffort"].values
    r2_raw = r2_score(y_true, y_pred)
    r2_log = r2_score(safe_log10(df["ActualEffort"]), safe_log10(df["PredictedEffort"]))

st.subheader("Model Validation")
cols = st.columns(2)
cols[0].metric("R² (Raw)", f"{r2_raw:.4f}")
cols[1].metric("R² (log10)", f"{r2_log:.4f}")

# ---------- Data preview ----------
st.subheader("Sample of Generated Dataset")
st.dataframe(df.head(20), use_container_width=True)

# ---------- Download ----------
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download Dataset (CSV)", csv_bytes, "pasta_effort_dataset.csv", "text/csv")
