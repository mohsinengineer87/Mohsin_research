
import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from sklearn.metrics import r2_score

# ---------- Page ----------
st.set_page_config(page_title="PASTA Effort Estimator", layout="wide")
st.title("📊 PASTA Effort Estimator — Noise, Log Mode & Sensitivity")

# ---------- Sidebar: Controls ----------
with st.sidebar:
    st.header("🔧 Dataset Settings")
    num_samples = st.slider("Number of Samples", 50, 20000, 200)
    scaling_factor = st.slider("Scaling Factor (S)", 0.1, 3.0, 1.1, 0.1)
    rng_seed = st.number_input("Random Seed", value=42)

    st.header("🏭 Industry Benchmarks")
    use_baselines = st.checkbox("Use Industry Standard Baselines for All Variables", value=False)

    industry_baseline = {
        "Assets": 4,
        "ThreatVectors": 8,
        "Vulnerabilities": 2100,
        "Complexity": 3.4,
        "ChangeRate": 3,
        "OrgMaturity": 3,
        "Automation": 2.3
    }

    st.header("🎛️ Attribute Ranges")

    def slider_or_baseline(label, key, minv, maxv):
        if use_baselines:
            st.info(f"{label}: Using Baseline = {industry_baseline[key]}")
            return (industry_baseline[key], industry_baseline[key])
        else:
            return st.slider(label, minv, maxv, (minv, maxv))

    asset_range = slider_or_baseline("Assets (A)", "Assets", 10, 10000)
    threat_range = slider_or_baseline("Threat Vectors (T)", "ThreatVectors", 1, 5000)
    vuln_range = slider_or_baseline("Vulnerabilities (V)", "Vulnerabilities", 1, 20000)
    complexity_range = slider_or_baseline("Complexity (C)", "Complexity", 1, 10)
    change_rate_range = slider_or_baseline("Change Rate (R)", "ChangeRate", 1, 10)
    maturity_range = slider_or_baseline("Org Maturity (M)", "OrgMaturity", 1, 10)
    automation_range = slider_or_baseline("Automation (Au)", "Automation", 1, 10)

    st.header("🔊 Noise Settings")
    noise_type = st.selectbox("Noise Distribution", ["None", "Gaussian", "Laplace"])
    noise_pct = st.slider("Noise Scale (% mean Predicted)", 0.0, 100.0, 5.0, 0.5)
    noise_seed = st.number_input("Noise Seed", value=123)

    st.header("📐 Analysis Settings")
    log_mode = st.checkbox("Analyze in log10 space", value=False)
    ofat_points = st.slider("OFAT sweep points", 10, 200, 50)
    selected_var = st.selectbox("Sensitivity Variable",
                                ["Assets", "ThreatVectors", "Vulnerabilities", "Complexity",
                                 "ChangeRate", "OrgMaturity", "Automation"])


# ---------- Helpers ----------
def generate_data(n, ranges, seed):
    np.random.seed(seed)
    A = np.random.randint(ranges["Assets"][0], ranges["Assets"][1] + 1, n)
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
    den = den.replace(0, 1)
    return (num.astype(float)) ** S / den

def add_noise(y_pred, noise_type, noise_pct, seed):
    if noise_type == "None" or noise_pct == 0:
        return y_pred.copy()
    rng = np.random.default_rng(seed)
    scale = (noise_pct / 100.0) * float(np.mean(y_pred))
    if noise_type == "Gaussian":
        noise = rng.normal(0.0, scale, len(y_pred))
    else:
        noise = rng.laplace(0.0, scale, len(y_pred))
    return y_pred + noise

def safe_log10(x):
    x = np.array(x, float)
    minpos = np.min(x[x > 0]) if np.any(x > 0) else 1.0
    return np.log10(np.clip(x, minpos * 1e-9, None))

def figure_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    buf.seek(0)
    return buf

# ---------- Generate data ----------
ranges = {
    "Assets": asset_range, "ThreatVectors": threat_range, "Vulnerabilities": vuln_range,
    "Complexity": complexity_range, "ChangeRate": change_rate_range,
    "OrgMaturity": maturity_range, "Automation": automation_range,
}

df = generate_data(num_samples, ranges, rng_seed)
df["PredictedEffort"] = model_predicted(df, scaling_factor)
df["ActualEffort"] = add_noise(df["PredictedEffort"], noise_type, noise_pct, noise_seed)

# ---------- R² ----------
if log_mode:
    y_true = safe_log10(df["ActualEffort"])
    y_pred = safe_log10(df["PredictedEffort"])
    r2_raw = r2_score(df["ActualEffort"], df["PredictedEffort"])
    r2_log = r2_score(y_true, y_pred)
else:
    y_true = df["ActualEffort"]
    y_pred = df["PredictedEffort"]
    r2_raw = r2_score(y_true, y_pred)
    r2_log = r2_score(safe_log10(df["ActualEffort"]), safe_log10(df["PredictedEffort"]))

st.subheader("Model Validation")
c1, c2 = st.columns(2)
c1.metric("R² (Raw)", f"{r2_raw:.4f}")
c2.metric("R² (log10)", f"{r2_log:.4f}")

# ---------- Data preview ----------
st.subheader("Sample Dataset")
st.dataframe(df.head(20))

st.download_button("Download Dataset (CSV)", df.to_csv(index=False).encode(),
                   "pasta_effort_dataset.csv", "text/csv")

# ---------- Plots ----------
pc1, pc2 = st.columns(2)

with pc1:
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.set_title("Actual vs Predicted")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with pc2:
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_true, y_true - y_pred, alpha=0.6)
    ax2.axhline(0, linestyle="--")
    ax2.set_title("Residuals")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

# ---------- Sensitivity (OFAT) ----------
st.subheader("Sensitivity Analysis (OFAT)")

med = {col: int(np.median(df[col])) for col in df.columns if col not in ["PredictedEffort", "ActualEffort"]}
low, high = ranges[selected_var]
xs = np.linspace(low, high, ofat_points)

rows = []
for x in xs:
    r = med.copy()
    r[selected_var] = x
    rows.append(r)

ofat_df = pd.DataFrame(rows)
ofat_df["PredictedEffort"] = model_predicted(ofat_df, scaling_factor)

fig3, ax3 = plt.subplots()
ax3.plot(xs, ofat_df["PredictedEffort"])
ax3.set_title(f"Effect of {selected_var}")
ax3.grid(True)
st.pyplot(fig3)
