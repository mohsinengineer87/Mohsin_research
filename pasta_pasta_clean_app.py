
import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from sklearn.metrics import r2_score

# ---------------- Page config ----------------
st.set_page_config(page_title="PASTA Effort Estimator (Clean)", layout="wide")
st.title("📊 PASTA Effort Estimator — Assets, CVSS & Core Metrics")

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("🔧 Dataset Settings")
    num_samples = st.slider("Number of Samples", 50, 20000, 500, help="How many rows to generate.")
    scaling_factor = st.slider("Scaling Factor (S)", 0.1, 3.0, 1.1, 0.1)
    rng_seed = st.number_input("Random Seed", value=42, help="For reproducibility.")

    st.header("🎛️ Attribute Ranges")

    # Assets: count + value
    asset_range = st.slider("Assets – Count (A)", 10, 20000, (100, 1000))
    use_asset_value = st.checkbox(
        "Use Asset Value (scale 1–100)",
        value=True,
        help="If enabled, use the Asset Value slider as a fixed value for all rows; "
             "if disabled, Asset Value is random between 1 and 100."
    )
    asset_value = st.slider(
        "Asset Value (1–100)",
        1, 100, 60,
        help="Represents business/impact value per asset on a 1–100 scale."
    )

    # Threat vectors
    threat_range = st.slider("Threat Vectors (T)", 1, 10000, (10, 200))

    # Vulnerabilities + CVSS
    vuln_range = st.slider("Vulnerabilities – Count (V)", 1, 50000, (50, 1000))
    use_cvss_baseline = st.checkbox(
        "Use CVSS / National Vulnerability Database (NIST) Baseline",
        value=True,
        help="If enabled, use a fixed average CVSS base score for all samples; "
             "if disabled, CVSS scores are random between 0.0 and 10.0."
    )
    cvss_baseline = st.slider(
        "Average Vulnerability Severity (CVSS Base Score 0–10)",
        0.0, 10.0, 7.5, 0.1,
        help="Represents the average CVSS base score (from NVD) for the vulnerabilities in scope."
    )

    # Remaining attributes
    complexity_range = st.slider("Complexity (C)", 1, 10, (2, 7))
    change_rate_range = st.slider("Change Rate (R)", 1, 10, (2, 7))
    maturity_range = st.slider("Org Maturity (M)", 1, 10, (2, 7))
    automation_range = st.slider("Automation (Au)", 1, 10, (2, 7))

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
        help="If enabled, metrics and plots use log10(Actual) vs log10(Predicted)."
    )

# ---------------- Helpers ----------------
def generate_data(
    n,
    ranges,
    seed,
    use_asset_value=True,
    asset_value=60,
    use_cvss_baseline=True,
    cvss_baseline=7.5,
):
    """
    Generate synthetic data for the PASTA effort model.
    - AssetsCount: sampled from range
    - AssetValue: fixed (slider) if use_asset_value else random 1–100
    - Vulnerabilities: count sampled from range
    - CVSSScore: fixed (slider) if use_cvss_baseline else random 0–10
    - Others: sampled from their ranges
    """
    np.random.seed(seed)

    # Asset count
    A_count = np.random.randint(ranges["AssetsCount"][0], ranges["AssetsCount"][1] + 1, n)

    # Asset value (1–100)
    if use_asset_value:
        A_val = np.full(n, float(asset_value))
    else:
        A_val = np.random.randint(1, 101, n).astype(float)

    # Threat vectors
    T = np.random.randint(ranges["ThreatVectors"][0], ranges["ThreatVectors"][1] + 1, n)

    # Vulnerabilities count
    V_count = np.random.randint(ranges["Vulnerabilities"][0], ranges["Vulnerabilities"][1] + 1, n)

    # CVSS score (0–10)
    if use_cvss_baseline:
        CVSS = np.full(n, float(cvss_baseline))
    else:
        CVSS = np.random.uniform(0.0, 10.0, n)

    # Complexity
    C = np.random.randint(ranges["Complexity"][0], ranges["Complexity"][1] + 1, n)
    # Change Rate
    R = np.random.randint(ranges["ChangeRate"][0], ranges["ChangeRate"][1] + 1, n)
    # Org Maturity
    M = np.random.randint(ranges["OrgMaturity"][0], ranges["OrgMaturity"][1] + 1, n)
    # Automation
    Au = np.random.randint(ranges["Automation"][0], ranges["Automation"][1] + 1, n)

    return pd.DataFrame({
        "AssetsCount": A_count,
        "AssetValue": A_val,
        "ThreatVectors": T,
        "Vulnerabilities": V_count,
        "CVSSScore": CVSS,
        "Complexity": C,
        "ChangeRate": R,
        "OrgMaturity": M,
        "Automation": Au,
    })

def model_predicted(df, S):
    """
    Effort model using:
    - Effective assets = AssetsCount × AssetValue
    - Effective vulnerabilities = Vulnerabilities × CVSSScore
    """
    eff_assets = df["AssetsCount"] * df["AssetValue"]
    eff_vulns = df["Vulnerabilities"] * df["CVSSScore"].clip(lower=0.1)  # avoid zero severity

    num = eff_assets * df["ThreatVectors"] * eff_vulns * df["Complexity"] * df["ChangeRate"]
    den = df["OrgMaturity"] * df["Automation"]
    den = den.replace(0, 1)  # defensive
    return (num.astype(np.float64)) ** S / den

def add_noise(y_pred, noise_type, noise_pct, seed):
    if noise_type == "None" or noise_pct == 0.0:
        return y_pred.copy()
    rng = np.random.default_rng(seed)
    scale = (noise_pct / 100.0) * float(np.mean(y_pred))
    if scale == 0:
        return y_pred.copy()
    if noise_type == "Gaussian":
        noise = rng.normal(loc=0.0, scale=scale, size=len(y_pred))
    else:  # Laplace
        noise = rng.laplace(loc=0.0, scale=scale, size=len(y_pred))
    return y_pred + noise

def safe_log10(x):
    x = np.asarray(x, dtype=np.float64)
    positive = x[x > 0]
    min_pos = np.min(positive) if positive.size > 0 else 1.0
    eps = min_pos * 1e-9
    return np.log10(np.clip(x, eps, None))

def figure_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    buf.seek(0)
    return buf

# ---------------- Generate data & targets ----------------
ranges = {
    "AssetsCount": asset_range,
    "ThreatVectors": threat_range,
    "Vulnerabilities": vuln_range,
    "Complexity": complexity_range,
    "ChangeRate": change_rate_range,
    "OrgMaturity": maturity_range,
    "Automation": automation_range,
}

df = generate_data(
    num_samples,
    ranges,
    seed=rng_seed,
    use_asset_value=use_asset_value,
    asset_value=asset_value,
    use_cvss_baseline=use_cvss_baseline,
    cvss_baseline=cvss_baseline,
)
df["PredictedEffort"] = model_predicted(df, scaling_factor)
df["ActualEffort"] = add_noise(df["PredictedEffort"], noise_type, noise_pct, seed=noise_seed)

# ---------------- Metrics ----------------
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

st.subheader("✅ Model Validation")
cols = st.columns(2)
cols[0].metric("R² (Raw)", f"{r2_raw:.4f}")
cols[1].metric("R² (log10)", f"{r2_log:.4f}")
st.caption("Synthetic targets (Actual = Predicted + noise) → optimistic R²; use as a structural check.")

# ---------------- Data preview & CSV download ----------------
st.subheader("📋 Sample of Generated Dataset")
st.dataframe(df.head(20), use_container_width=True)

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Dataset (CSV)", csv_bytes,
                   file_name="pasta_effort_dataset.csv", mime="text/csv")

config = {
    "num_samples": num_samples,
    "scaling_factor_S": scaling_factor,
    "ranges": {k: list(v) for k, v in ranges.items()},
    "rng_seed": rng_seed,
    "asset": {"use_asset_value": use_asset_value, "asset_value": asset_value},
    "vulnerabilities": {"use_cvss_baseline": use_cvss_baseline, "cvss_baseline": cvss_baseline},
    "noise": {"type": noise_type, "percent_of_mean": noise_pct, "noise_seed": noise_seed},
    "analysis": {"log_mode": log_mode},
}
st.download_button("🧾 Download Config (JSON)", json.dumps(config, indent=2).encode("utf-8"),
                   file_name="experiment_config.json", mime="application/json")

st.markdown("---")

# ---------------- Plots ----------------
plot_col1, plot_col2 = st.columns(2)

with plot_col1:
    st.markdown("### Actual vs Predicted")
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.scatter(y_true, y_pred, alpha=0.6)
    xymin = float(min(np.min(y_true), np.min(y_pred)))
    xymax = float(max(np.max(y_true), np.max(y_pred)))
    ax1.plot([xymin, xymax], [xymin, xymax], linestyle="--")
    ax1.set_xlabel("Actual" + (" (log10)" if log_mode else ""))
    ax1.set_ylabel("Predicted" + (" (log10)" if log_mode else ""))
    ax1.set_title("Actual vs Predicted Effort")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1, use_container_width=True)
    st.download_button("📤 Download Plot: Actual vs Predicted (PNG)",
                       data=figure_to_bytes(fig1),
                       file_name="actual_vs_predicted.png", mime="image/png")

with plot_col2:
    st.markdown("### Residuals")
    residuals = y_true - y_pred
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.scatter(y_true, residuals, alpha=0.6)
    ax2.axhline(0.0, linestyle="--")
    ax2.set_xlabel("Actual" + (" (log10)" if log_mode else ""))
    ax2.set_ylabel("Residual (Actual - Predicted)" + (" (log10 space)" if log_mode else ""))
    ax2.set_title("Residuals vs Actual")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2, use_container_width=True)
    st.download_button("📤 Download Plot: Residuals (PNG)",
                       data=figure_to_bytes(fig2),
                       file_name="residuals.png", mime="image/png")

st.markdown("---")

st.markdown(r"""
### Model Interpretation

Effective assets:  
\(
A_{\text{eff}} = A_{\text{count}} \times A_{\text{value}}
\)

Effective vulnerabilities:  
\(
V_{\text{eff}} = V_{\text{count}} \times \text{CVSS}
\)

Effort:  
\(
\hat{E} = \frac{(A_{\text{eff}} \times T \times V_{\text{eff}} \times C \times R)^{S}}{M \times Au}
\)

where  
- \(A_{\text{count}}\)=AssetsCount, \(A_{\text{value}}\)=AssetValue (1–100),  
- \(T\)=ThreatVectors, \(V_{\text{count}}\)=Vulnerabilities, \(\text{CVSS}\)=CVSSScore,  
- \(C\)=Complexity, \(R\)=ChangeRate, \(M\)=OrgMaturity, \(Au\)=Automation,  
- \(S\)=Scaling factor.
""")
