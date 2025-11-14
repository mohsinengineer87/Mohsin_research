# app.py
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
    num_samples = st.slider("Number of Samples", 50, 20000, 200, help="How many rows to generate.")
    scaling_factor = st.slider("Scaling Factor (S)", 0.1, 3.0, 1.1, 0.1)
    rng_seed = st.number_input("Random Seed", value=42, help="For reproducibility.")

    st.header("🎛️ Attribute Ranges")
    asset_range = st.slider("Assets (A)", 10, 10000, (100, 300))
    # Asset value sub-toggle (scale 1–100)
    use_asset_value_baseline = st.checkbox(
        "Use Asset Value (scale 1–100)",
        value=False,
        help="If enabled, all samples use the fixed Asset Value below; otherwise a random 1–100 value is used per sample."
    )
    asset_value = st.slider(
        "Asset Value (1–100)",
        1, 100, 50,
        help="Represents the business/impact value of assets on a 1–100 scale."
    )
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
def generate_data(n, ranges, seed, use_asset_value_baseline=False, asset_value=None):
    np.random.seed(seed)
    # Asset count
    A = np.random.randint(ranges["Assets"][0], ranges["Assets"][1] + 1, n)

    # Asset value (1–100 scale)
    if use_asset_value_baseline and asset_value is not None:
        Av = np.full(n, int(asset_value), dtype=int)
    else:
        Av = np.random.randint(1, 101, n)

    T = np.random.randint(ranges["ThreatVectors"][0], ranges["ThreatVectors"][1] + 1, n)
    V = np.random.randint(ranges["Vulnerabilities"][0], ranges["Vulnerabilities"][1] + 1, n)
    C = np.random.randint(ranges["Complexity"][0], ranges["Complexity"][1] + 1, n)
    R = np.random.randint(ranges["ChangeRate"][0], ranges["ChangeRate"][1] + 1, n)
    M = np.random.randint(ranges["OrgMaturity"][0], ranges["OrgMaturity"][1] + 1, n)
    Au = np.random.randint(ranges["Automation"][0], ranges["Automation"][1] + 1, n)
    return pd.DataFrame({
        "Assets": A,
        "AssetValue": Av,
        "ThreatVectors": T,
        "Vulnerabilities": V,
        "Complexity": C,
        "ChangeRate": R,
        "OrgMaturity": M,
        "Automation": Au
    })

def model_predicted(df, S):
    # Incorporate both asset count and asset value (1–100) into the asset term
    asset_term = df["Assets"] * df.get("AssetValue", 1)
    num = asset_term * df["ThreatVectors"] * df["Vulnerabilities"] * df["Complexity"] * df["ChangeRate"]
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

df = generate_data(num_samples, ranges, seed=rng_seed,
                     use_asset_value_baseline=use_asset_value_baseline,
                     asset_value=asset_value)
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

st.subheader("✅ Model Validation")
cols = st.columns(2)
cols[0].metric("R² (Raw)", f"{r2_raw:.4f}")
cols[1].metric("R² (log10)", f"{r2_log:.4f}")
st.caption("Note: Synthetic targets (Actual = Predicted + small noise) yield optimistic R²; use as a sanity check.")

# ---------- Data preview & downloads ----------
st.subheader("📋 Sample of Generated Dataset")
st.dataframe(df.head(20), use_container_width=True)

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Dataset (CSV)", csv_bytes, file_name="pasta_effort_dataset.csv", mime="text/csv")

config = {
    "num_samples": num_samples, "scaling_factor_S": scaling_factor,
    "ranges": ranges, "rng_seed": rng_seed,
    "noise": {"type": noise_type, "percent_of_mean": noise_pct, "noise_seed": noise_seed},
    "analysis": {"log_mode": log_mode, "ofat_points": ofat_points, "selected_var": selected_var},
}
st.download_button("🧾 Download Config (JSON)", json.dumps(config, indent=2).encode("utf-8"),
                   file_name="experiment_config.json", mime="application/json")

st.markdown("---")

# ---------- Plots: Actual vs Predicted + Residuals ----------
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

# ---------- Sensitivity Analysis ----------
st.subheader("📈 Sensitivity Analysis (OFAT + Elasticities)")
st.caption("One-Factor-At-A-Time (OFAT): sweep one variable across its range; hold others at their medians.")

medians = {
    "Assets": int(np.median(df["Assets"])),
    "AssetValue": int(np.median(df["AssetValue"])),
    "ThreatVectors": int(np.median(df["ThreatVectors"])),
    "Vulnerabilities": int(np.median(df["Vulnerabilities"])),
    "Complexity": int(np.median(df["Complexity"])),
    "ChangeRate": int(np.median(df["ChangeRate"])),
    "OrgMaturity": int(np.median(df["OrgMaturity"])),
    "Automation": int(np.median(df["Automation"])),
}

lo, hi = ranges[selected_var]
xs = np.linspace(lo, hi, ofat_points, dtype=float)

base = medians.copy()
ofat_rows = []
for v in xs:
    row = base.copy()
    if selected_var in ["Assets", "ThreatVectors", "Vulnerabilities", "OrgMaturity", "Automation"]:
        row[selected_var] = int(round(v))
    else:
        row[selected_var] = float(v)
    ofat_rows.append(row)

ofat_df = pd.DataFrame(ofat_rows)
ofat_df["PredictedEffort"] = model_predicted(ofat_df, scaling_factor)

y_ofat = np.log10(ofat_df["PredictedEffort"]) if log_mode else ofat_df["PredictedEffort"]

fig3, ax3 = plt.subplots(figsize=(7, 4.8))
ax3.plot(xs, y_ofat)
ax3.set_title(f"OFAT: Effect of {selected_var} on Predicted Effort" + (" (log10)" if log_mode else ""))
ax3.set_xlabel(f"{selected_var}")
ax3.set_ylabel("Predicted Effort" + (" (log10)" if log_mode else ""))
ax3.grid(True, alpha=0.3)
st.pyplot(fig3, use_container_width=True)
st.download_button("📤 Download Plot: Sensitivity (PNG)",
                   data=figure_to_bytes(fig3),
                   file_name=f"sensitivity_{selected_var}.png", mime="image/png")

# ---------- Elasticities ----------
st.markdown("#### Elasticities at Median Operating Point")
st.caption("Closed-form (from the model) and empirical (finite difference) elasticities.")

closed_form = {
    "Assets": scaling_factor, "ThreatVectors": scaling_factor, "Vulnerabilities": scaling_factor,
    "Complexity": scaling_factor, "ChangeRate": scaling_factor,
    "OrgMaturity": -1.0, "Automation": -1.0
}

def empirical_elasticity(var, base_row, rel_delta=0.05):
    x0 = max(1e-9, float(base_row[var]))
    step = max(1.0, abs(x0 * rel_delta)) if var in ["Assets","ThreatVectors","Vulnerabilities","OrgMaturity","Automation"] else x0 * rel_delta
    row0, rowp = base_row.copy(), base_row.copy()
    rowp[var] = type(x0)(x0 + step)

    E0 = float(model_predicted(pd.DataFrame([row0]), scaling_factor).iloc[0])
    Ep = float(model_predicted(pd.DataFrame([rowp]), scaling_factor).iloc[0])

    dE_over_E = (Ep - E0) / max(E0, 1e-12)
    dX_over_X = (rowp[var] - x0) / max(x0, 1e-12)
    return dE_over_E / dX_over_X if dX_over_X != 0 else np.nan

empirical = {v: empirical_elasticity(v, medians) for v in closed_form.keys()}

elas_df = pd.DataFrame({
    "Variable": list(closed_form.keys()),
    "ClosedForm_Elasticity": list(closed_form.values()),
    "Empirical_Elasticity": [empirical[v] for v in closed_form.keys()]
})
st.dataframe(elas_df, use_container_width=True)
st.download_button("📥 Download Elasticities (CSV)", elas_df.to_csv(index=False).encode("utf-8"),
                   "elasticities.csv", "text/csv")

st.markdown("---")

st.markdown(r"""
**Model Formula**  
\[
\hat{E} = \frac{(A \times T \times V \times C \times R)^{S}}{M \times Au}
\]
where  
- \(A\)=Assets, \(T\)=ThreatVectors, \(V\)=Vulnerabilities, \(C\)=Complexity, \(R\)=ChangeRate,  
- \(M\)=OrgMaturity, \(Au\)=Automation, \(S\)=Scaling factor.
""")
