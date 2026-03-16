"""
PASTA Scalability Threat Modeling — Streamlit App
Research Focus: Quantitative Scalability Evaluation of PASTA Framework
"""

# ── IMPORTS ────────────────────────────────────────────────────────────────────
import io
import json
import time
import tracemalloc

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

matplotlib.use("Agg")  # non-interactive backend — required for Streamlit

# ── PAGE CONFIG (MUST be the absolute first Streamlit call) ────────────────────
st.set_page_config(
    page_title="PASTA Scalability Estimator",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SESSION STATE DEFAULTS ────────────────────────────────────────────────────
_defaults = {
    "T_weight": 1.0,
    "benchmark_results": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── THREAT VECTOR TAXONOMY ────────────────────────────────────────────────────
THREAT_VECTORS = {
    "MITRE_ATT&CK": {
        "Initial Access": ["Phishing", "Drive-by Compromise", "Valid Accounts"],
        "Execution": ["Command and Scripting Interpreter", "PowerShell"],
        "Persistence": [
            "Registry Run Keys/Startup Folder",
            "Boot or Logon Autostart Execution",
        ],
    },
    "ENISA_Threat_Landscape": {
        "Top Threats": [
            "Ransomware", "Phishing", "Malware",
            "Data Breaches", "Denial of Service", "Insider Threats",
        ]
    },
    "VERIS": {
        "Actor": ["External", "Internal", "Partner"],
        "Action": ["Malware", "Hacking", "Social", "Misuse"],
        "Asset": ["Server", "Person", "User Device"],
        "Attribute": ["Confidentiality", "Integrity", "Availability"],
    },
}

MITRE_TECHNIQUE_WEIGHTS = {
    "Phishing (T1566.001)": 1.5,
    "Drive-by Compromise (T1189)": 1.3,
    "Valid Accounts (T1078)": 1.2,
    "Command-Line Interface (T1059.003)": 1.4,
    "PowerShell (T1059.001)": 1.6,
    "Registry Run Keys (T1547.001)": 1.3,
    "Boot or Logon Autostart Execution (T1547)": 1.2,
    "Scheduled Task (T1053.005)": 1.1,
    "Token Impersonation (T1134.001)": 1.4,
    "Bypass UAC (T1548.002)": 1.35,
    "Exfiltration Over Web (T1041)": 1.25,
}

# PASTA 7-Stage labels (for display/documentation)
PASTA_STAGES = {
    1: "Define Objectives",
    2: "Define Technical Scope",
    3: "Decompose Application",
    4: "Analyze Threats",
    5: "Identify Vulnerabilities",
    6: "Enumerate Attacks",
    7: "Analyze Risk & Impact",
}

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔐 PASTA Controls")

    # --- MITRE technique selection (moved into sidebar to avoid collision) ---
    st.header("🧠 MITRE ATT&CK Techniques")
    mitre_selected = st.multiselect(
        "Select techniques:",
        list(MITRE_TECHNIQUE_WEIGHTS.keys()),
        default=["Phishing (T1566.001)", "PowerShell (T1059.001)"],
        key="sidebar_mitre_techniques",
    )
    T_weight = sum(MITRE_TECHNIQUE_WEIGHTS.get(t, 1.0) for t in mitre_selected)
    T_weight = T_weight if T_weight > 0 else 1.0
    st.session_state["T_weight"] = T_weight
    st.metric("Threat Vector Multiplier (T_weight)", f"{T_weight:.2f}")

    st.divider()
    st.header("⚙️ Dataset Settings")
    num_samples = st.slider("Number of Samples", 50, 20_000, 500,
                            help="Rows in the synthetic dataset.")
    scaling_factor = st.slider("Scaling Factor (S)", 0.1, 3.0, 1.1, 0.1)
    rng_seed = st.number_input("Random Seed", value=42)

    st.header("🎛️ Attribute Ranges")
    asset_range = st.slider("Assets – Count (A)", 10, 20_000, (100, 1_000))
    use_asset_value = st.checkbox("Fixed Asset Value", value=True,
                                  help="Use slider value for all rows; otherwise random 1–100.")
    asset_value = st.slider("Asset Value (1–100)", 1, 100, 60)

    threat_range = st.slider("Threat Vectors (T)", 1, 10_000, (10, 200))
    vuln_range = st.slider("Vulnerabilities – Count (V)", 1, 50_000, (50, 1_000))
    use_cvss_baseline = st.checkbox("Fixed CVSS Baseline", value=True,
                                    help="Use slider value for all rows; otherwise random 0–10.")
    cvss_baseline = st.slider("Avg CVSS Base Score (0–10)", 0.0, 10.0, 7.5, 0.1)

    complexity_range = st.slider("Complexity (C)", 1, 10, (2, 7))
    change_rate_range = st.slider("Change Rate (R)", 1, 10, (2, 7))
    maturity_range = st.slider("Org Maturity (M)", 1, 10, (2, 7))
    automation_range = st.slider("Automation (Au)", 1, 10, (2, 7))

    st.header("🔊 Noise Settings")
    noise_type = st.selectbox("Noise Distribution", ["None", "Gaussian", "Laplace"])
    noise_pct = st.slider("Noise Scale (% of mean Predicted)", 0.0, 100.0, 5.0, 0.5)
    noise_seed = st.number_input("Noise Seed", value=123)

    st.header("📐 Analysis")
    log_mode = st.checkbox("Plot / Metrics in log₁₀ space", value=False)
    ofat_points = st.slider("OFAT Sweep Points", 10, 200, 50)
    selected_var = st.selectbox(
        "Sensitivity Variable (OFAT)",
        ["AssetsCount", "AssetValue", "ThreatVectors", "Vulnerabilities",
         "CVSSScore", "Complexity", "ChangeRate", "OrgMaturity", "Automation"],
    )


# ── HELPER FUNCTIONS ───────────────────────────────────────────────────────────
def figure_to_bytes(fig: plt.Figure) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf


def safe_log10(x):
    x = np.asarray(x, dtype=np.float64)
    positive = x[x > 0]
    eps = (np.min(positive) * 1e-9) if positive.size > 0 else 1e-12
    return np.log10(np.clip(x, eps, None))


# ── DATA GENERATION (cached) ──────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_data(
    n: int,
    ranges: dict,
    seed: int,
    use_asset_value: bool,
    asset_value: float,
    use_cvss_baseline: bool,
    cvss_baseline: float,
) -> pd.DataFrame:
    """Generate synthetic PASTA threat-modeling dataset."""
    rng = np.random.default_rng(seed)

    A_count = rng.integers(ranges["AssetsCount"][0], ranges["AssetsCount"][1] + 1, n)
    A_val = np.full(n, float(asset_value)) if use_asset_value else rng.integers(1, 101, n).astype(float)
    T = rng.integers(ranges["ThreatVectors"][0], ranges["ThreatVectors"][1] + 1, n)
    V_count = rng.integers(ranges["Vulnerabilities"][0], ranges["Vulnerabilities"][1] + 1, n)
    CVSS = np.full(n, float(cvss_baseline)) if use_cvss_baseline else rng.uniform(0.0, 10.0, n)
    C = rng.integers(ranges["Complexity"][0], ranges["Complexity"][1] + 1, n)
    R = rng.integers(ranges["ChangeRate"][0], ranges["ChangeRate"][1] + 1, n)
    M = rng.integers(ranges["OrgMaturity"][0], ranges["OrgMaturity"][1] + 1, n)
    Au = rng.integers(ranges["Automation"][0], ranges["Automation"][1] + 1, n)

    return pd.DataFrame({
        "AssetsCount": A_count, "AssetValue": A_val,
        "ThreatVectors": T, "Vulnerabilities": V_count, "CVSSScore": CVSS,
        "Complexity": C, "ChangeRate": R, "OrgMaturity": M, "Automation": Au,
    })


def model_predicted(df: pd.DataFrame, S: float, T_weight: float = 1.0) -> pd.Series:
    """
    PASTA Effort Model:
        E = ( A_eff × T × T_weight × V_eff × C × R )^S  /  (M × Au)
    where A_eff = AssetsCount × AssetValue,  V_eff = Vulnerabilities × CVSS
    """
    eff_assets = df["AssetsCount"] * df["AssetValue"]
    eff_vulns = df["Vulnerabilities"] * df["CVSSScore"].clip(lower=0.1)
    num = eff_assets * df["ThreatVectors"] * T_weight * eff_vulns * df["Complexity"] * df["ChangeRate"]
    den = (df["OrgMaturity"] * df["Automation"]).replace(0, 1)
    return (num.astype(np.float64)) ** S / den


def add_noise(y_pred: pd.Series, noise_type: str, noise_pct: float, seed: int) -> np.ndarray:
    if noise_type == "None" or noise_pct == 0.0:
        return y_pred.values.copy()
    rng = np.random.default_rng(seed)
    scale = (noise_pct / 100.0) * float(np.mean(y_pred))
    if scale == 0:
        return y_pred.values.copy()
    noise = rng.normal(0, scale, len(y_pred)) if noise_type == "Gaussian" \
        else rng.laplace(0, scale, len(y_pred))
    return y_pred.values + noise


def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_log = r2_score(safe_log10(y_true), safe_log10(y_pred))
    return {"R²": r2, "R² (log₁₀)": r2_log, "MAE": mae, "RMSE": rmse}


# ── SCALABILITY BENCHMARK (cached) ───────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_scalability_benchmark(
    sample_sizes: tuple,
    ranges: dict,
    seed: int,
    S: float,
    T_weight: float,
    use_asset_value: bool,
    asset_value: float,
    use_cvss_baseline: bool,
    cvss_baseline: float,
) -> pd.DataFrame:
    """
    Measure wall-clock time, peak memory, and throughput for data generation
    and effort prediction across increasing problem sizes.
    Returns a DataFrame with scalability metrics.
    """
    records = []
    for n in sample_sizes:
        # --- Data generation ---
        tracemalloc.start()
        t0 = time.perf_counter()
        df_bench = generate_data(n, ranges, seed, use_asset_value, asset_value,
                                 use_cvss_baseline, cvss_baseline)
        gen_time = time.perf_counter() - t0
        _, gen_mem_peak = tracemalloc.get_traced_memory()  # bytes
        tracemalloc.stop()

        # --- Prediction / feature engineering ---
        tracemalloc.start()
        t1 = time.perf_counter()
        y_pred_bench = model_predicted(df_bench, S, T_weight)
        pred_time = time.perf_counter() - t1
        _, pred_mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_time = gen_time + pred_time
        throughput = n / total_time if total_time > 0 else float("inf")

        records.append({
            "N (Samples)": n,
            "Data Gen Time (s)": round(gen_time, 4),
            "Prediction Time (s)": round(pred_time, 4),
            "Total Time (s)": round(total_time, 4),
            "Peak Mem – Data Gen (KB)": round(gen_mem_peak / 1024, 2),
            "Peak Mem – Prediction (KB)": round(pred_mem_peak / 1024, 2),
            "Throughput (samples/s)": round(throughput, 1),
        })

    return pd.DataFrame(records)


# ── MAIN DATASET BUILD ────────────────────────────────────────────────────────
ranges = {
    "AssetsCount": asset_range, "ThreatVectors": threat_range,
    "Vulnerabilities": vuln_range, "Complexity": complexity_range,
    "ChangeRate": change_rate_range, "OrgMaturity": maturity_range,
    "Automation": automation_range,
}

with st.spinner("Generating dataset…"):
    df = generate_data(
        num_samples, ranges, rng_seed, use_asset_value, asset_value,
        use_cvss_baseline, cvss_baseline,
    )

df["PredictedEffort"] = model_predicted(df, scaling_factor, T_weight=st.session_state["T_weight"])
df["ActualEffort"] = add_noise(df["PredictedEffort"], noise_type, noise_pct, noise_seed)

y_true = safe_log10(df["ActualEffort"]) if log_mode else df["ActualEffort"].values
y_pred = safe_log10(df["PredictedEffort"]) if log_mode else df["PredictedEffort"].values
metrics = compute_metrics(df["ActualEffort"].values, df["PredictedEffort"].values)

# ── APP TITLE ─────────────────────────────────────────────────────────────────
st.title("🔐 PASTA Scalability Estimator")
st.caption(
    "Quantitative scalability evaluation of the PASTA (Process for Attack Simulation & "
    "Threat Analysis) framework — measuring computation time, memory usage, and throughput "
    "across synthetic threat-modeling pipelines."
)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_overview, tab_data, tab_plots, tab_sensitivity, tab_scalability, tab_framework = st.tabs([
    "📋 Overview",
    "📊 Dataset & Metrics",
    "📈 Plots",
    "🔬 Sensitivity (OFAT)",
    "⚡ Scalability Benchmarks",
    "🗺️ Framework Reference",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    st.subheader("PASTA 7-Stage Framework")
    stage_cols = st.columns(7)
    colors = ["#1f4e79", "#2e75b6", "#2980b9", "#1abc9c",
              "#e67e22", "#c0392b", "#8e44ad"]
    for i, (stage_num, stage_name) in enumerate(PASTA_STAGES.items()):
        stage_cols[i].markdown(
            f"""<div style='background:{colors[i]};border-radius:8px;padding:10px;
            text-align:center;color:white;font-size:0.78rem;'>
            <strong>Stage {stage_num}</strong><br>{stage_name}</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.subheader("📐 Effort Model")
    st.markdown(r"""
The PASTA effort model estimates threat-modeling workload as:

$$
\hat{E} = \frac{(A_{\text{count}} \times A_{\text{value}} \times T \times T_w \times V_{\text{count}} \times \text{CVSS} \times C \times R)^{S}}{M \times Au}
$$

| Symbol | Meaning | Scalability Stage |
|--------|---------|-------------------|
| $A_{\text{count}}$ | Asset count | Stage 2 – Technical Scope |
| $A_{\text{value}}$ | Asset business value (1–100) | Stage 7 – Risk/Impact |
| $T$ | Threat vector count | Stage 4 – Threat Analysis |
| $T_w$ | MITRE technique weight multiplier | Stage 4 |
| $V_{\text{count}}$ | Vulnerability count | Stage 5 – Vulnerability ID |
| CVSS | Avg vulnerability severity (NVD) | Stage 5 |
| $C$ | System complexity | Stage 3 – Decomposition |
| $R$ | Change rate | Stage 3 |
| $M$ | Org security maturity (reduces effort) | Stage 1 – Objectives |
| $Au$ | Automation level (reduces effort) | Stage 1 |
| $S$ | Scaling exponent | — |

> **Elasticity note**: All numerator variables have elasticity $= S$ (doubling any raises effort by $2^S$).
> Denominator variables (M, Au) have elasticity $= -1$ (doubling halves effort).
    """)

    st.subheader("🎯 Current Configuration Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples (N)", f"{num_samples:,}")
    c2.metric("Scaling Factor (S)", scaling_factor)
    c3.metric("T_weight (MITRE)", f"{st.session_state['T_weight']:.2f}")
    c4.metric("Noise Type", noise_type)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — DATASET & METRICS
# ─────────────────────────────────────────────────────────────────────────────
with tab_data:
    st.subheader("✅ Model Validation Metrics")
    mc = st.columns(4)
    mc[0].metric("R²", f"{metrics['R²']:.4f}")
    mc[1].metric("R² (log₁₀)", f"{metrics['R² (log₁₀)']:.4f}")
    mc[2].metric("MAE", f"{metrics['MAE']:.2e}")
    mc[3].metric("RMSE", f"{metrics['RMSE']:.2e}")
    st.caption(
        "Synthetic targets (Actual = Predicted + noise) → near-perfect R² is expected; "
        "use MAE/RMSE to assess noise magnitude relative to signal."
    )

    st.subheader("📋 Generated Dataset Preview (first 20 rows)")
    st.dataframe(df.head(20), use_container_width=True)

    c_dl1, c_dl2 = st.columns(2)
    with c_dl1:
        st.download_button(
            "📥 Download Full Dataset (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="pasta_effort_dataset.csv",
            mime="text/csv",
        )

    config_payload = {
        "num_samples": num_samples,
        "scaling_factor_S": scaling_factor,
        "ranges": {k: list(v) for k, v in ranges.items()},
        "rng_seed": rng_seed,
        "asset": {"use_asset_value": use_asset_value, "asset_value": asset_value},
        "vulnerabilities": {"use_cvss_baseline": use_cvss_baseline, "cvss_baseline": cvss_baseline},
        "noise": {"type": noise_type, "percent_of_mean": noise_pct, "noise_seed": noise_seed},
        "mitre_selected": mitre_selected,
        "T_weight": st.session_state["T_weight"],
        "analysis": {"log_mode": log_mode, "ofat_points": ofat_points, "selected_var": selected_var},
    }
    with c_dl2:
        st.download_button(
            "🧾 Download Config (JSON)",
            json.dumps(config_payload, indent=2).encode("utf-8"),
            file_name="experiment_config.json",
            mime="application/json",
        )

    st.subheader("📊 Dataset Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — PLOTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_plots:
    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        st.markdown("### Actual vs Predicted Effort")
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.scatter(y_true, y_pred, alpha=0.4, s=8, color="#2e75b6")
        lim = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
        ax1.plot(lim, lim, "r--", linewidth=1.5, label="Ideal")
        ax1.set_xlabel("Actual" + (" (log₁₀)" if log_mode else ""))
        ax1.set_ylabel("Predicted" + (" (log₁₀)" if log_mode else ""))
        ax1.set_title("Actual vs Predicted Effort")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1, use_container_width=True)
        st.download_button("📤 Download (PNG)", figure_to_bytes(fig1),
                           "actual_vs_predicted.png", "image/png")
        plt.close(fig1)

    with plot_col2:
        st.markdown("### Residuals")
        residuals = df["ActualEffort"].values - df["PredictedEffort"].values
        if log_mode:
            residuals = safe_log10(df["ActualEffort"]) - safe_log10(df["PredictedEffort"])
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.scatter(y_true, residuals, alpha=0.4, s=8, color="#c0392b")
        ax2.axhline(0.0, linestyle="--", color="black", linewidth=1)
        ax2.set_xlabel("Actual" + (" (log₁₀)" if log_mode else ""))
        ax2.set_ylabel("Residual (Actual − Predicted)" + (" log₁₀ space" if log_mode else ""))
        ax2.set_title("Residuals vs Actual")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2, use_container_width=True)
        st.download_button("📤 Download (PNG)", figure_to_bytes(fig2),
                           "residuals.png", "image/png")
        plt.close(fig2)

    # Distribution of predicted effort
    st.markdown("### Distribution of Predicted Effort")
    fig3, ax3 = plt.subplots(figsize=(9, 3.5))
    data_plot = safe_log10(df["PredictedEffort"]) if log_mode else df["PredictedEffort"]
    ax3.hist(data_plot, bins=60, color="#1abc9c", edgecolor="white", linewidth=0.3)
    ax3.set_xlabel("Predicted Effort" + (" (log₁₀)" if log_mode else ""))
    ax3.set_ylabel("Frequency")
    ax3.set_title("Distribution of Predicted Effort")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3, use_container_width=True)
    st.download_button("📤 Download (PNG)", figure_to_bytes(fig3),
                       "effort_distribution.png", "image/png")
    plt.close(fig3)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — SENSITIVITY (OFAT)
# ─────────────────────────────────────────────────────────────────────────────
with tab_sensitivity:
    st.subheader("📈 One-Factor-At-A-Time (OFAT) Sensitivity Analysis")
    st.caption("Sweep one variable across its full range; hold all others at their dataset medians.")

    medians = {col: float(np.median(df[col])) for col in
               ["AssetsCount", "AssetValue", "ThreatVectors", "Vulnerabilities",
                "CVSSScore", "Complexity", "ChangeRate", "OrgMaturity", "Automation"]}

    sweep_bounds = {
        "AssetsCount": ranges["AssetsCount"],
        "ThreatVectors": ranges["ThreatVectors"],
        "Vulnerabilities": ranges["Vulnerabilities"],
        "Complexity": ranges["Complexity"],
        "ChangeRate": ranges["ChangeRate"],
        "OrgMaturity": ranges["OrgMaturity"],
        "Automation": ranges["Automation"],
        "AssetValue": (1.0, 100.0),
        "CVSSScore": (0.0, 10.0),
    }
    lo, hi = sweep_bounds[selected_var]
    xs = np.linspace(lo, hi, ofat_points, dtype=float)
    integer_vars = {"AssetsCount", "ThreatVectors", "Vulnerabilities",
                    "Complexity", "ChangeRate", "OrgMaturity", "Automation"}

    ofat_rows = []
    for v in xs:
        row = medians.copy()
        row[selected_var] = float(int(round(v))) if selected_var in integer_vars else float(v)
        ofat_rows.append(row)
    ofat_df = pd.DataFrame(ofat_rows)
    ofat_df["PredictedEffort"] = model_predicted(ofat_df, scaling_factor,
                                                  T_weight=st.session_state["T_weight"])
    y_ofat = safe_log10(ofat_df["PredictedEffort"]) if log_mode else ofat_df["PredictedEffort"]

    fig4, ax4 = plt.subplots(figsize=(8, 4.5))
    ax4.plot(xs, y_ofat, linewidth=2, color="#8e44ad")
    ax4.set_title(f"OFAT: Effect of {selected_var} on Predicted Effort" +
                  (" (log₁₀)" if log_mode else ""))
    ax4.set_xlabel(selected_var)
    ax4.set_ylabel("Predicted Effort" + (" (log₁₀)" if log_mode else ""))
    ax4.grid(True, alpha=0.3)
    st.pyplot(fig4, use_container_width=True)
    st.download_button("📤 Download Plot (PNG)", figure_to_bytes(fig4),
                       f"sensitivity_{selected_var}.png", "image/png")
    plt.close(fig4)

    ofat_export = ofat_df.copy()
    ofat_export[selected_var] = xs
    cols_ordered = [selected_var] + [c for c in ofat_export.columns if c != selected_var]
    st.download_button(
        "📥 Download OFAT Data (CSV)",
        ofat_export[cols_ordered].to_csv(index=False).encode("utf-8"),
        file_name=f"ofat_{selected_var}.csv", mime="text/csv",
    )

    # Elasticities
    st.subheader("📊 Elasticities at Median Operating Point")
    st.caption("Closed-form (from model structure) vs empirical (finite-difference) elasticities.")

    closed_form = {
        "AssetsCount": scaling_factor, "AssetValue": scaling_factor,
        "ThreatVectors": scaling_factor, "Vulnerabilities": scaling_factor,
        "CVSSScore": scaling_factor, "Complexity": scaling_factor,
        "ChangeRate": scaling_factor, "OrgMaturity": -1.0, "Automation": -1.0,
    }

    def empirical_elasticity(var, base_row, rel_delta=0.05):
        x0 = max(1e-9, float(base_row[var]))
        step = max(1.0, abs(x0 * rel_delta)) if var in integer_vars else (x0 * rel_delta or rel_delta)
        x1 = x0 + step
        r0 = base_row.copy(); r1 = base_row.copy(); r1[var] = x1
        E0 = float(model_predicted(pd.DataFrame([r0]), scaling_factor).iloc[0])
        E1 = float(model_predicted(pd.DataFrame([r1]), scaling_factor).iloc[0])
        dE_E = (E1 - E0) / max(E0, 1e-12)
        dX_X = (x1 - x0) / max(x0, 1e-12)
        return dE_E / dX_X if dX_X != 0 else np.nan

    empirical = {v: empirical_elasticity(v, medians) for v in closed_form}
    elas_df = pd.DataFrame({
        "Variable": list(closed_form.keys()),
        "ClosedForm_Elasticity": list(closed_form.values()),
        "Empirical_Elasticity": [empirical[v] for v in closed_form],
        "Interpretation": [
            "Scales with S (numerator)" if v not in {"OrgMaturity", "Automation"}
            else "Reduces effort (denominator)"
            for v in closed_form
        ],
    })
    st.dataframe(elas_df, use_container_width=True)
    st.download_button("📥 Download Elasticities (CSV)",
                       elas_df.to_csv(index=False).encode("utf-8"),
                       "elasticities.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — SCALABILITY BENCHMARKS  (core research contribution)
# ─────────────────────────────────────────────────────────────────────────────
with tab_scalability:
    st.subheader("⚡ Quantitative Scalability Benchmarks")
    st.caption(
        "Measures wall-clock time, peak memory, and throughput as N (samples / problem size) "
        "grows. This directly evaluates the scalability hypothesis of the research problem statement."
    )

    bench_col1, bench_col2 = st.columns([2, 1])
    with bench_col1:
        n_points_bench = st.slider(
            "Number of benchmark sizes to evaluate", 5, 20, 10,
            help="Evenly-spaced N values between min and max below."
        )
    with bench_col2:
        bench_max = st.number_input("Max N for benchmark", value=10_000, step=1_000,
                                    min_value=500, max_value=50_000)

    bench_sizes = tuple(
        int(x) for x in np.linspace(100, bench_max, n_points_bench, dtype=int)
    )

    run_bench = st.button("▶ Run Scalability Benchmark", type="primary")

    if run_bench:
        with st.spinner(f"Benchmarking {len(bench_sizes)} problem sizes …"):
            bench_df = run_scalability_benchmark(
                bench_sizes, ranges, rng_seed, scaling_factor,
                st.session_state["T_weight"], use_asset_value, asset_value,
                use_cvss_baseline, cvss_baseline,
            )
        st.session_state["benchmark_results"] = bench_df

    if st.session_state["benchmark_results"] is not None:
        bench_df = st.session_state["benchmark_results"]

        st.subheader("📋 Benchmark Results Table")
        st.dataframe(bench_df.style.format({
            "Data Gen Time (s)": "{:.4f}",
            "Prediction Time (s)": "{:.4f}",
            "Total Time (s)": "{:.4f}",
            "Peak Mem – Data Gen (KB)": "{:.1f}",
            "Peak Mem – Prediction (KB)": "{:.1f}",
            "Throughput (samples/s)": "{:,.0f}",
        }), use_container_width=True)

        st.download_button(
            "📥 Download Benchmark Data (CSV)",
            bench_df.to_csv(index=False).encode("utf-8"),
            "pasta_scalability_benchmark.csv", "text/csv",
        )

        # ── 4-panel scalability plot ──────────────────────────────────────────
        st.subheader("📈 Scalability Profiles")
        fig_b, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig_b.suptitle("PASTA Pipeline Scalability vs Problem Size (N)", fontsize=13, fontweight="bold")
        Ns = bench_df["N (Samples)"]

        # 1. Total time
        axes[0, 0].plot(Ns, bench_df["Total Time (s)"], marker="o", color="#2e75b6")
        axes[0, 0].set_title("Total Wall-Clock Time")
        axes[0, 0].set_xlabel("N (Samples)")
        axes[0, 0].set_ylabel("Seconds")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Time breakdown (stacked area)
        axes[0, 1].stackplot(
            Ns,
            bench_df["Data Gen Time (s)"],
            bench_df["Prediction Time (s)"],
            labels=["Data Generation", "Effort Prediction"],
            colors=["#1abc9c", "#e67e22"], alpha=0.8,
        )
        axes[0, 1].set_title("Time Breakdown by Stage")
        axes[0, 1].set_xlabel("N (Samples)")
        axes[0, 1].set_ylabel("Seconds")
        axes[0, 1].legend(loc="upper left", fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Memory usage
        axes[1, 0].plot(Ns, bench_df["Peak Mem – Data Gen (KB)"], marker="s",
                        label="Data Gen", color="#8e44ad")
        axes[1, 0].plot(Ns, bench_df["Peak Mem – Prediction (KB)"], marker="^",
                        label="Prediction", color="#c0392b")
        axes[1, 0].set_title("Peak Memory Usage")
        axes[1, 0].set_xlabel("N (Samples)")
        axes[1, 0].set_ylabel("KB")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Throughput
        axes[1, 1].bar(Ns, bench_df["Throughput (samples/s)"],
                       color="#2980b9", width=(Ns.iloc[-1] - Ns.iloc[0]) / len(Ns) * 0.7)
        axes[1, 1].set_title("Throughput")
        axes[1, 1].set_xlabel("N (Samples)")
        axes[1, 1].set_ylabel("Samples / Second")
        axes[1, 1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        st.pyplot(fig_b, use_container_width=True)
        st.download_button("📤 Download Benchmark Plots (PNG)", figure_to_bytes(fig_b),
                           "pasta_scalability_plots.png", "image/png")
        plt.close(fig_b)

        # ── Scalability complexity estimate ─────────────────────────────────
        st.subheader("📐 Empirical Complexity Estimate")
        st.caption("Log-log slope of Total Time vs N estimates the empirical time complexity O(N^k).")
        log_N = np.log(bench_df["N (Samples)"].values.astype(float))
        log_T = np.log(bench_df["Total Time (s)"].values.astype(float) + 1e-9)
        slope, intercept = np.polyfit(log_N, log_T, 1)
        st.markdown(
            f"**Fitted slope k = {slope:.3f}** → empirical complexity ≈ **O(N^{slope:.2f})**  \n"
            f"A slope near 1.0 indicates linear scaling; above 1.5 suggests super-linear bottlenecks."
        )

    else:
        st.info("👆 Click **Run Scalability Benchmark** above to generate empirical measurements.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — FRAMEWORK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
with tab_framework:
    st.subheader("🗺️ PASTA Framework Reference")

    with st.expander("📌 PASTA 7-Stage Methodology", expanded=True):
        for s_num, s_name in PASTA_STAGES.items():
            st.markdown(f"**Stage {s_num} — {s_name}**")

        st.markdown("""
        **Scalability Bottlenecks by Stage:**
        | Stage | Scalability Concern |
        |-------|---------------------|
        | 1 – Define Objectives | Manual effort; doesn't scale with system size |
        | 2 – Technical Scope | Asset inventory explosion in large systems |
        | 3 – Decompose Application | Graph complexity grows O(N²) for dense architectures |
        | 4 – Analyze Threats | Threat vector space expands combinatorially |
        | 5 – Identify Vulnerabilities | CVE database queries scale with asset × vuln count |
        | 6 – Enumerate Attacks | Attack path enumeration is NP-hard in general |
        | 7 – Risk & Impact | Score aggregation scales linearly but data prep doesn't |
        """)

    with st.expander("🔐 Threat Vector Taxonomy (MITRE / ENISA / VERIS)"):
        st.markdown("**MITRE ATT&CK:**")
        for tactic, techniques in THREAT_VECTORS["MITRE_ATT&CK"].items():
            st.markdown(f"- **{tactic}**: {', '.join(techniques)}")
        st.markdown("**ENISA Threat Landscape:**")
        for t in THREAT_VECTORS["ENISA_Threat_Landscape"]["Top Threats"]:
            st.markdown(f"- {t}")
        st.markdown("**VERIS Schema:**")
        for category, items in THREAT_VECTORS["VERIS"].items():
            st.markdown(f"- **{category}**: {', '.join(items)}")

    with st.expander("🎯 Interactive Threat Vector Explorer (by Standard)"):
        standard = st.radio("Security Standard:", ("MITRE ATT&CK", "ENISA", "VERIS"),
                            horizontal=True)
        if standard == "MITRE ATT&CK":
            exp_techniques = st.multiselect(
                "Select techniques:",
                list(MITRE_TECHNIQUE_WEIGHTS.keys()),
                key="framework_mitre",
            )
            if exp_techniques:
                w = sum(MITRE_TECHNIQUE_WEIGHTS[t] for t in exp_techniques)
                st.info(f"Combined T_weight for selection: **{w:.2f}** "
                        f"(use sidebar selection to apply to model)")
        elif standard == "ENISA":
            exp_enisa = st.multiselect(
                "Select ENISA threats:",
                THREAT_VECTORS["ENISA_Threat_Landscape"]["Top Threats"],
                key="framework_enisa",
            )
            if exp_enisa:
                st.info(f"Selected {len(exp_enisa)} ENISA threat(s).")
        else:
            exp_veris = st.multiselect(
                "Select VERIS actions:",
                THREAT_VECTORS["VERIS"]["Action"],
                key="framework_veris",
            )
            if exp_veris:
                st.info(f"Selected VERIS actions: {', '.join(exp_veris)}")

    st.subheader("📚 Key References")
    st.markdown("""
    - **PASTA**: Tony UcedaVélez & Marco M. Morana, *Risk Centric Threat Modeling*, Wiley 2015
    - **MITRE ATT&CK**: https://attack.mitre.org
    - **ENISA Threat Landscape**: https://www.enisa.europa.eu/topics/cyber-threats/enisa-threat-landscape
    - **VERIS**: http://veriscommunity.net
    - **NVD / CVSS**: https://nvd.nist.gov/vuln-metrics/cvss
    """)
