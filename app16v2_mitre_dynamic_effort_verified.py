
import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from sklearn.metrics import r2_score
# ===============================
# 📌 Threat Vectors Mapping (For Scalability and Threat Attribution)
# ===============================
threat_vectors = {
    "MITRE_ATT&CK": {
        "Initial Access": ["Phishing", "Drive-by Compromise", "Valid Accounts"],
        "Execution": ["Command and Scripting Interpreter", "PowerShell"],
        "Persistence": ["Registry Run Keys/Startup Folder", "Boot or Logon Autostart Execution"]
    },
    "ENISA_Threat_Landscape": {
        "Top Threats": [
            "Ransomware",
            "Phishing",
            "Malware",
            "Data Breaches",
            "Denial of Service",
            "Insider Threats"
        ]
    },
    "VERIS": {
        "Actor": ["External", "Internal", "Partner"],
        "Action": ["Malware", "Hacking", "Social", "Misuse"],
        "Asset": ["Server", "Person", "User Device"],
        "Attribute": ["Confidentiality", "Integrity", "Availability"]
    }
}



# === MITRE Threat Technique Integration ===
st.header("📌 MITRE Threat Techniques")
mitre_techniques = [
    "Phishing", "Drive-by Compromise", "Valid Accounts",
    "Command and Scripting Interpreter", "PowerShell",
    "Registry Run Keys/Startup Folder", "Boot or Logon Autostart Execution"
]

technique_weights = {
    "Phishing": 1.5,
    "Drive-by Compromise": 1.3,
    "Valid Accounts": 1.2,
    "Command and Scripting Interpreter": 1.4,
    "PowerShell": 1.6,
    "Registry Run Keys/Startup Folder": 1.3,
    "Boot or Logon Autostart Execution": 1.2
}

selected_techniques = st.multiselect("Select MITRE Techniques:", mitre_techniques)
T_weight = sum(technique_weights.get(t, 1.0) for t in selected_techniques)
st.write(f"🧠 Threat Vector Multiplier (T_weight): {T_weight}")


# ---------------- Page config ----------------
st.set_page_config(page_title="PASTA Effort Estimator (Clean)", layout="wide")
st.title("📊 PASTA Effort Estimator — Assets, CVSS, Sensitivity & Metrics")

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
    ofat_points = st.slider(
        "OFAT sweep points", 10, 200, 50,
        help="Number of points in one-factor-at-a-time sensitivity sweep."
    )
    selected_var = st.selectbox(
        "Sensitivity variable (OFAT)",
        [
            "AssetsCount",
            "AssetValue",
            "ThreatVectors",
            "Vulnerabilities",
            "CVSSScore",
            "Complexity",
            "ChangeRate",
            "OrgMaturity",
            "Automation",
        ],
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

def model_predicted(df, S, T_weight):
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

# ---------------- Refresh Predicted & Actual Effort Dynamically ----------------
df = df.copy()
df["PredictedEffort"] = model_predicted(df, scaling_factor, T_weight)
df["PredictedEffort_NoThreats"] = model_predicted(df, scaling_factor, 1.0)
df["EffortDelta"] = df["PredictedEffort"] - df["PredictedEffort_NoThreats"]
df["ActualEffort"] = add_noise(df["PredictedEffort"], noise_type, noise_pct, seed=noise_seed)

st.subheader("📊 MITRE Impact on Predicted Effort")
st.metric("📉 Predicted Effort (row 0)", round(df["PredictedEffort"].iloc[0], 2))
st.metric("📉 Predicted Effort (No Threats)", round(df["PredictedEffort_NoThreats"].iloc[0], 2))
st.metric("📈 Delta Due to Threats", round(df["EffortDelta"].iloc[0], 2))

st.subheader("📋 Updated Effort Table (Top 5 Rows)")
st.dataframe(df[["PredictedEffort_NoThreats", "PredictedEffort", "EffortDelta", "ActualEffort"]].head())

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
    "analysis": {"log_mode": log_mode, "ofat_points": ofat_points, "selected_var": selected_var},
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

# ---------------- Sensitivity Analysis (OFAT) ----------------
st.subheader("📈 Sensitivity Analysis (OFAT)")
st.caption("One-Factor-At-A-Time (OFAT): sweep one variable across its range; hold others at their medians.")

medians = {
    "AssetsCount": float(np.median(df["AssetsCount"])),
    "AssetValue": float(np.median(df["AssetValue"])),
    "ThreatVectors": float(np.median(df["ThreatVectors"])),
    "Vulnerabilities": float(np.median(df["Vulnerabilities"])),
    "CVSSScore": float(np.median(df["CVSSScore"])),
    "Complexity": float(np.median(df["Complexity"])),
    "ChangeRate": float(np.median(df["ChangeRate"])),
    "OrgMaturity": float(np.median(df["OrgMaturity"])),
    "Automation": float(np.median(df["Automation"])),
}

# decide sweep range
if selected_var in ["AssetsCount", "ThreatVectors", "Vulnerabilities",
                    "Complexity", "ChangeRate", "OrgMaturity", "Automation"]:
    lo, hi = ranges[selected_var]
elif selected_var == "AssetValue":
    lo, hi = 1.0, 100.0
elif selected_var == "CVSSScore":
    lo, hi = 0.0, 10.0
else:
    lo, hi = 0.0, 1.0

xs = np.linspace(lo, hi, ofat_points, dtype=float)

base = medians.copy()
ofat_rows = []
for v in xs:
    row = base.copy()
    if selected_var in ["AssetsCount", "ThreatVectors", "Vulnerabilities",
                        "Complexity", "ChangeRate", "OrgMaturity", "Automation"]:
        row[selected_var] = float(int(round(v)))
    else:
        row[selected_var] = float(v)
    ofat_rows.append(row)

ofat_df = pd.DataFrame(ofat_rows)
ofat_df["PredictedEffort"] = model_predicted(ofat_df, scaling_factor, T_weight)
y_ofat = safe_log10(ofat_df["PredictedEffort"]) if log_mode else ofat_df["PredictedEffort"]

fig3, ax3 = plt.subplots(figsize=(7, 4.5))
ax3.plot(xs, y_ofat)
ax3.set_title(f"OFAT: Effect of {selected_var} on Predicted Effort" + (" (log10)" if log_mode else ""))
ax3.set_xlabel(selected_var)
ax3.set_ylabel("Predicted Effort" + (" (log10)" if log_mode else ""))
ax3.grid(True, alpha=0.3)
st.pyplot(fig3, use_container_width=True)
st.download_button("📤 Download Plot: Sensitivity (PNG)",
                   data=figure_to_bytes(fig3),
                   file_name=f"sensitivity_{selected_var}.png", mime="image/png")

ofat_df_out = ofat_df.copy()
ofat_df_out[selected_var] = xs
cols = [selected_var] + [col for col in ofat_df_out.columns if col != selected_var]
ofat_df_out = ofat_df_out[cols]
st.download_button(
    "📥 Download OFAT Data (CSV)",
    ofat_df_out.to_csv(index=False).encode("utf-8"),
    file_name=f"ofat_{selected_var}.csv",
    mime="text/csv",
)

# ---------------- Elasticities ----------------
st.subheader("📊 Elasticities at Median Operating Point")
st.caption("Closed-form (from model structure) and empirical (finite difference) elasticities.")

closed_form = {
    "AssetsCount": scaling_factor,
    "AssetValue": scaling_factor,
    "ThreatVectors": scaling_factor,
    "Vulnerabilities": scaling_factor,
    "CVSSScore": scaling_factor,
    "Complexity": scaling_factor,
    "ChangeRate": scaling_factor,
    "OrgMaturity": -1.0,
    "Automation": -1.0,
}

def empirical_elasticity(var, base_row, rel_delta=0.05):
    x0 = max(1e-9, float(base_row[var]))
    # integer-like vs continuous
    if var in ["AssetsCount", "ThreatVectors", "Vulnerabilities",
               "Complexity", "ChangeRate", "OrgMaturity", "Automation"]:
        step = max(1.0, abs(x0 * rel_delta))
        x1 = x0 + step
    else:  # AssetValue, CVSSScore
        step = x0 * rel_delta
        if step == 0:
            step = rel_delta
        x1 = x0 + step

    row0 = base_row.copy()
    row1 = base_row.copy()
    row1[var] = x1

    E0 = float(model_predicted(pd.DataFrame([row0]), scaling_factor).iloc[0])
    E1 = float(model_predicted(pd.DataFrame([row1]), scaling_factor).iloc[0])

    dE_over_E = (E1 - E0) / max(E0, 1e-12)
    dX_over_X = (x1 - x0) / max(x0, 1e-12)
    if dX_over_X == 0:
        return np.nan
    return dE_over_E / dX_over_X

empirical = {v: empirical_elasticity(v, medians) for v in closed_form.keys()}

elas_df = pd.DataFrame({
    "Variable": list(closed_form.keys()),
    "ClosedForm_Elasticity": list(closed_form.values()),
    "Empirical_Elasticity": [empirical[v] for v in closed_form.keys()],
})
st.dataframe(elas_df, use_container_width=True)
st.download_button(
    "📥 Download Elasticities (CSV)",
    elas_df.to_csv(index=False).encode("utf-8"),
    file_name="elasticities.csv",
    mime="text/csv",
)

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

# ===============================
# 🛡️ Threat Vectors Dashboard Panel (Linked to Scalability)
# ===============================
with st.expander('🔐 View Threat Vector Mapping (MITRE / ENISA / VERIS)'):
    st.markdown('**MITRE ATT&CK:**')
    for tactic, techniques in threat_vectors['MITRE_ATT&CK'].items():
        st.markdown(f'- **{tactic}**: {", ".join(techniques)}')
    st.markdown('**ENISA Threat Landscape (Top Threats):**')
    for threat in threat_vectors['ENISA_Threat_Landscape']['Top Threats']:
        st.markdown(f'- {threat}')
    st.markdown('**VERIS Schema:**')
    for category, items in threat_vectors['VERIS'].items():
        st.markdown(f'- **{category}**: {", ".join(items)}')

# ===============================
# 🎯 Threat Vectors by Security Standard (MITRE Subcategory)
# ===============================
with st.expander('🎯 Threat Vectors by Security Standard'):
    selected_standard = st.radio(
        'Choose Security Standard:',
        ('MITRE ATT&CK', 'ENISA Threat Landscape', 'VERIS')
    )
    if selected_standard == 'MITRE ATT&CK':
        selected_techniques = st.multiselect(
            'Select MITRE Techniques:',
            [
                'Phishing (T1566.001)',
                'PowerShell (T1059.001)',
                'Registry Run Keys (T1547.001)',
                'Scheduled Task (T1053.005)',
                'Command-Line Interface (T1059.003)',
                'Token Impersonation (T1134.001)',
                'Bypass UAC (T1548.002)',
                'Exfiltration Over Web (T1041)'
            ]
        )
        T = len(selected_techniques) if selected_techniques else 1
        st.write('Selected MITRE Techniques:', selected_techniques)
    elif selected_standard == 'ENISA Threat Landscape':
        selected_enisa = st.multiselect(
            'Select ENISA Threats:',
            ['Ransomware', 'Phishing', 'Data Breaches', 'Insider Threats', 'Denial of Service']
        )
        T = len(selected_enisa) if selected_enisa else 1
        st.write('Selected ENISA Threats:', selected_enisa)
    elif selected_standard == 'VERIS':
        selected_veris = st.multiselect(
            'Select VERIS Actions:',
            ['Malware', 'Hacking', 'Social', 'Misuse', 'Physical', 'Error']
        )
        T = len(selected_veris) if selected_veris else 1
        st.write('Selected VERIS Actions:', selected_veris)



# ---------------- Dynamic Effort Update Block ----------------
df = df.copy()

# ---------------- Refresh Predicted & Actual Effort Dynamically ----------------
df = df.copy()
df["PredictedEffort"] = model_predicted(df, scaling_factor, T_weight)
df["PredictedEffort_NoThreats"] = model_predicted(df, scaling_factor, 1.0)
df["EffortDelta"] = df["PredictedEffort"] - df["PredictedEffort_NoThreats"]
df["ActualEffort"] = add_noise(df["PredictedEffort"], noise_type, noise_pct, seed=noise_seed)

st.subheader("📊 MITRE Impact on Predicted Effort")
st.metric("📉 Predicted Effort (row 0)", round(df["PredictedEffort"].iloc[0], 2))
st.metric("📉 Predicted Effort (No Threats)", round(df["PredictedEffort_NoThreats"].iloc[0], 2))
st.metric("📈 Delta Due to Threats", round(df["EffortDelta"].iloc[0], 2))

st.subheader("📋 Updated Effort Table (Top 5 Rows)")
st.dataframe(df[["PredictedEffort_NoThreats", "PredictedEffort", "EffortDelta", "ActualEffort"]].head())

df["ActualEffort"] = add_noise(df["PredictedEffort"], noise_type, noise_pct, seed=noise_seed)

st.subheader("📈 Updated Effort Metrics")
st.metric("📊 Predicted Effort (row 0)", round(df["PredictedEffort"].iloc[0], 2))
st.metric("🧪 Actual Effort (row 0)", round(df["ActualEffort"].iloc[0], 2))

st.subheader("📊 Full Effort Table (Top 5 Rows)")
st.dataframe(df[["PredictedEffort", "ActualEffort"]].head())
