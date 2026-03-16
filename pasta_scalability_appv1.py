"""
PASTA Scalability Threat Modeling — Interactive Streamlit App
Research Focus: Quantitative Scalability Evaluation of PASTA Framework
Audience: Academic / Research Reviewers

Run with:  streamlit run pasta_scalability_app.py
Requires:  pip install streamlit plotly pandas numpy scikit-learn
"""

import io, json, time, tracemalloc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── PAGE CONFIG  (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="PASTA Scalability Research Tool",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
.formula-box{background:#1e2530;border-left:4px solid #3b9dd2;border-radius:6px;
padding:14px 18px;font-family:monospace;font-size:1.02rem;color:#e8f4fd;margin:6px 0;}
.callout{background:#eaf4fb;border-left:4px solid #2980b9;padding:10px 16px;
border-radius:4px;font-size:0.9rem;margin:8px 0;}
.stage-pill{display:inline-block;padding:5px 12px;border-radius:16px;color:white;
font-size:0.8rem;font-weight:600;margin:3px;}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ───────────────────────────────────────────────────────────────
PASTA_STAGES = {
    1:{"name":"Define Objectives",       "icon":"🎯","color":"#1a5276",
       "model_vars":["OrgMaturity (M)"],
       "description":"Establish business objectives, security requirements, and risk appetite.",
       "scalability_concern":"Manual stakeholder elicitation — doesn't scale with system size.",
       "bottleneck_class":"O(1) per iteration but high human-hours cost",
       "research_note":"OrgMaturity (M) inversely moderates effort: higher maturity reduces per-objective cost."},
    2:{"name":"Define Technical Scope",  "icon":"🗺️","color":"#1f618d",
       "model_vars":["AssetsCount (A_count)","AssetValue (A_val)"],
       "description":"Enumerate infrastructure components, data flows, trust boundaries.",
       "scalability_concern":"Asset inventory explosion: O(N) storage, O(N²) dependency mapping.",
       "bottleneck_class":"O(N) — scales linearly with asset count",
       "research_note":"AssetsCount is the primary scaling driver at this stage."},
    3:{"name":"Decompose Application",   "icon":"🔩","color":"#2874a6",
       "model_vars":["Complexity (C)","ChangeRate (R)"],
       "description":"Decompose the system into components, data flows, entry/exit points.",
       "scalability_concern":"DFD complexity grows O(N²) for fully-connected architectures.",
       "bottleneck_class":"O(N²) worst-case for graph decomposition",
       "research_note":"Complexity (C) captures structural density; ChangeRate (R) models churn."},
    4:{"name":"Analyze Threats",         "icon":"⚔️","color":"#17a589",
       "model_vars":["ThreatVectors (T)","T_weight (MITRE)"],
       "description":"Identify threat agents, motives, and attack vectors using MITRE ATT&CK.",
       "scalability_concern":"Threat vector space expands combinatorially with assets × techniques.",
       "bottleneck_class":"O(A × T) — combinatorial expansion",
       "research_note":"T_weight captures technique sophistication. Without automation, this becomes dominant."},
    5:{"name":"Identify Vulnerabilities","icon":"🔍","color":"#d68910",
       "model_vars":["Vulnerabilities (V)","CVSSScore"],
       "description":"Map CVEs and weaknesses to system components using NVD/CVSS scores.",
       "scalability_concern":"CVE lookups scale O(V × A); CVSS scoring requires per-entry analysis.",
       "bottleneck_class":"O(V × A) — product of assets and vulnerabilities",
       "research_note":"CVSSScore weights V_count by severity. Most data-intensive stage."},
    6:{"name":"Enumerate Attacks",       "icon":"🕸️","color":"#ba4a00",
       "model_vars":["ThreatVectors (T)","Vulnerabilities (V)"],
       "description":"Build attack trees connecting threat agents to assets via vulnerabilities.",
       "scalability_concern":"Attack path enumeration is NP-hard in the general case.",
       "bottleneck_class":"NP-hard — exponential in worst case",
       "research_note":"Primary scalability bottleneck. Even with pruning, search space grows exponentially."},
    7:{"name":"Analyze Risk & Impact",   "icon":"📊","color":"#7d3c98",
       "model_vars":["AssetValue (A_val)","CVSSScore","OrgMaturity (M)","Automation (Au)"],
       "description":"Quantify residual risk combining attack likelihood with business impact.",
       "scalability_concern":"Score aggregation is O(N) but report generation is expensive.",
       "bottleneck_class":"O(N) with high constant — parallelizable",
       "research_note":"Automation (Au) and OrgMaturity (M) are denominators: mature orgs absorb effort sub-linearly."},
}

MITRE_TECHNIQUES = {
    "Phishing (T1566.001)":           {"weight":1.5,"tactic":"Initial Access","severity":"High"},
    "Drive-by Compromise (T1189)":    {"weight":1.3,"tactic":"Initial Access","severity":"Medium"},
    "Valid Accounts (T1078)":         {"weight":1.2,"tactic":"Initial Access","severity":"Medium"},
    "Command-Line (T1059.003)":       {"weight":1.4,"tactic":"Execution",     "severity":"High"},
    "PowerShell (T1059.001)":         {"weight":1.6,"tactic":"Execution",     "severity":"High"},
    "Registry Run Keys (T1547.001)":  {"weight":1.3,"tactic":"Persistence",   "severity":"Medium"},
    "Boot Autostart (T1547)":         {"weight":1.2,"tactic":"Persistence",   "severity":"Medium"},
    "Scheduled Task (T1053.005)":     {"weight":1.1,"tactic":"Persistence",   "severity":"Low"},
    "Token Impersonation (T1134.001)":{"weight":1.4,"tactic":"Privilege Esc.","severity":"High"},
    "Bypass UAC (T1548.002)":         {"weight":1.35,"tactic":"Privilege Esc.","severity":"High"},
    "Exfil Over Web (T1041)":         {"weight":1.25,"tactic":"Exfiltration", "severity":"Medium"},
}

ENISA_THREATS = {
    "Ransomware":       {"weight":1.7,"category":"Malware",     "trend":"↑ Increasing"},
    "Phishing":         {"weight":1.5,"category":"Social Eng.", "trend":"↑ Increasing"},
    "Malware":          {"weight":1.4,"category":"Malware",     "trend":"→ Stable"},
    "Data Breaches":    {"weight":1.3,"category":"Data",        "trend":"↑ Increasing"},
    "Denial of Service":{"weight":1.2,"category":"Availability","trend":"↑ Increasing"},
    "Insider Threats":  {"weight":1.1,"category":"Human",       "trend":"→ Stable"},
    "Supply Chain":     {"weight":1.6,"category":"Third-party", "trend":"↑ Increasing"},
    "Zero-day Exploits":{"weight":1.8,"category":"Exploitation","trend":"↑ Increasing"},
}

VERIS_ACTIONS = {
    "Malware":  {"weight":1.4,"actor":"External","asset":"Server"},
    "Hacking":  {"weight":1.5,"actor":"External","asset":"Server"},
    "Social":   {"weight":1.3,"actor":"External","asset":"Person"},
    "Misuse":   {"weight":1.2,"actor":"Internal","asset":"User Device"},
    "Physical": {"weight":1.1,"actor":"Internal","asset":"Physical"},
    "Error":    {"weight":1.0,"actor":"Internal","asset":"Person"},
}

# ── SESSION STATE ────────────────────────────────────────────────────────────
for k,v in {"T_weight":1.0,"benchmark_results":None,"walkthrough_stage":1}.items():
    if k not in st.session_state: st.session_state[k]=v

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔐 PASTA Controls")
    st.caption("All charts update live as you adjust parameters.")

    st.markdown("### 🧠 MITRE ATT&CK")
    mitre_selected = st.multiselect("Active techniques:",list(MITRE_TECHNIQUES.keys()),
        default=["Phishing (T1566.001)","PowerShell (T1059.001)"],key="sidebar_mitre")
    T_weight = sum(MITRE_TECHNIQUES[t]["weight"] for t in mitre_selected) or 1.0
    st.session_state["T_weight"] = T_weight
    st.metric("T_weight Multiplier",f"{T_weight:.2f}",delta=f"{T_weight-1:+.2f} vs baseline")

    st.divider()
    st.markdown("### ⚙️ Dataset")
    num_samples    = st.slider("Samples (N)",50,20_000,500)
    scaling_factor = st.slider("Scaling Factor (S)",0.1,3.0,1.1,0.1)
    rng_seed       = st.number_input("Random Seed",value=42,step=1)

    st.markdown("### 🎛️ Variable Ranges")
    asset_range       = st.slider("Asset Count (A)",10,20_000,(100,1_000))
    use_asset_value   = st.checkbox("Fixed Asset Value",value=True)
    asset_value       = st.slider("Asset Value (1–100)",1,100,60)
    threat_range      = st.slider("Threat Vectors (T)",1,10_000,(10,200))
    vuln_range        = st.slider("Vulnerabilities (V)",1,50_000,(50,1_000))
    use_cvss_baseline = st.checkbox("Fixed CVSS Baseline",value=True)
    cvss_baseline     = st.slider("Avg CVSS Score (0–10)",0.0,10.0,7.5,0.1)
    complexity_range  = st.slider("Complexity (C)",1,10,(2,7))
    change_rate_range = st.slider("Change Rate (R)",1,10,(2,7))
    maturity_range    = st.slider("Org Maturity (M)",1,10,(2,7))
    automation_range  = st.slider("Automation (Au)",1,10,(2,7))

    st.markdown("### 🔊 Noise")
    noise_type = st.selectbox("Distribution",["None","Gaussian","Laplace"])
    noise_pct  = st.slider("Noise Scale (%)",0.0,100.0,5.0,0.5)
    noise_seed = st.number_input("Noise Seed",value=123,step=1)

    st.markdown("### 📐 Analysis")
    log_mode    = st.checkbox("Use log₁₀ space",value=False)
    ofat_points = st.slider("OFAT Points",10,200,60)
    selected_var = st.selectbox("OFAT Variable",
        ["AssetsCount","AssetValue","ThreatVectors","Vulnerabilities",
         "CVSSScore","Complexity","ChangeRate","OrgMaturity","Automation"])

# ── CORE FUNCTIONS ───────────────────────────────────────────────────────────
def safe_log10(x):
    x = np.asarray(x,dtype=np.float64)
    pos = x[x>0]; eps = (np.min(pos)*1e-9) if pos.size>0 else 1e-12
    return np.log10(np.clip(x,eps,None))

@st.cache_data(show_spinner=False)
def generate_data(n,ranges,seed,use_av,av,use_cv,cv):
    rng = np.random.default_rng(seed)
    A  = rng.integers(ranges["AssetsCount"][0],   ranges["AssetsCount"][1]+1,   n)
    Av = np.full(n,float(av)) if use_av else rng.integers(1,101,n).astype(float)
    T  = rng.integers(ranges["ThreatVectors"][0], ranges["ThreatVectors"][1]+1, n)
    V  = rng.integers(ranges["Vulnerabilities"][0],ranges["Vulnerabilities"][1]+1,n)
    CV = np.full(n,float(cv)) if use_cv else rng.uniform(0,10,n)
    C  = rng.integers(ranges["Complexity"][0],    ranges["Complexity"][1]+1,    n)
    R  = rng.integers(ranges["ChangeRate"][0],    ranges["ChangeRate"][1]+1,    n)
    M  = rng.integers(ranges["OrgMaturity"][0],   ranges["OrgMaturity"][1]+1,   n)
    Au = rng.integers(ranges["Automation"][0],    ranges["Automation"][1]+1,    n)
    return pd.DataFrame({"AssetsCount":A,"AssetValue":Av,"ThreatVectors":T,
        "Vulnerabilities":V,"CVSSScore":CV,"Complexity":C,"ChangeRate":R,"OrgMaturity":M,"Automation":Au})

def model_predicted(df,S,Tw=1.0):
    eA  = df["AssetsCount"]*df["AssetValue"]
    eV  = df["Vulnerabilities"]*df["CVSSScore"].clip(lower=0.1)
    num = eA*df["ThreatVectors"]*Tw*eV*df["Complexity"]*df["ChangeRate"]
    den = (df["OrgMaturity"]*df["Automation"]).replace(0,1)
    return (num.astype(np.float64))**S/den

def add_noise(y,nt,pct,seed):
    if nt=="None" or pct==0: return y.values.copy()
    rng=np.random.default_rng(seed); sc=(pct/100)*float(np.mean(y))
    if sc==0: return y.values.copy()
    return y.values+(rng.normal(0,sc,len(y)) if nt=="Gaussian" else rng.laplace(0,sc,len(y)))

def compute_metrics(yt,yp):
    return {"R²":r2_score(yt,yp),"R² log₁₀":r2_score(safe_log10(yt),safe_log10(yp)),
            "MAE":mean_absolute_error(yt,yp),"RMSE":float(np.sqrt(mean_squared_error(yt,yp))),
            "MAPE (%)":float(np.mean(np.abs((yt-yp)/np.clip(np.abs(yt),1e-9,None)))*100)}

@st.cache_data(show_spinner=False)
def run_scalability_benchmark(sizes,ranges,seed,S,Tw,use_av,av,use_cv,cv):
    records=[]
    for n in sizes:
        tracemalloc.start(); t0=time.perf_counter()
        db=generate_data(n,ranges,seed,use_av,av,use_cv,cv)
        gt=time.perf_counter()-t0; _,gm=tracemalloc.get_traced_memory(); tracemalloc.stop()
        tracemalloc.start(); t1=time.perf_counter()
        model_predicted(db,S,Tw)
        pt=time.perf_counter()-t1; _,pm=tracemalloc.get_traced_memory(); tracemalloc.stop()
        tot=gt+pt
        records.append({"N":n,"Gen Time (s)":round(gt,5),"Pred Time (s)":round(pt,5),
            "Total (s)":round(tot,5),"Gen Mem (KB)":round(gm/1024,2),
            "Pred Mem (KB)":round(pm/1024,2),"Throughput":round(n/tot if tot>0 else 1e9,1)})
    return pd.DataFrame(records)

# ── BUILD DATASET ────────────────────────────────────────────────────────────
ranges = {"AssetsCount":asset_range,"ThreatVectors":threat_range,"Vulnerabilities":vuln_range,
          "Complexity":complexity_range,"ChangeRate":change_rate_range,
          "OrgMaturity":maturity_range,"Automation":automation_range}

with st.spinner("Generating dataset…"):
    df = generate_data(num_samples,ranges,rng_seed,use_asset_value,asset_value,
                       use_cvss_baseline,cvss_baseline)

df["PredictedEffort"] = model_predicted(df,scaling_factor,st.session_state["T_weight"])
df["ActualEffort"]    = add_noise(df["PredictedEffort"],noise_type,noise_pct,noise_seed)
metrics               = compute_metrics(df["ActualEffort"].values,df["PredictedEffort"].values)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("# 🔐 PASTA Scalability Research Tool")
st.caption("Interactive quantitative evaluation of the PASTA framework's scalability. "
           "All charts update live as sidebar parameters change.")

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Samples (N)",f"{num_samples:,}")
k2.metric("Scaling Factor",f"S = {scaling_factor}")
k3.metric("T_weight",f"{st.session_state['T_weight']:.2f}")
k4.metric("R²",f"{metrics['R²']:.4f}")
k5.metric("RMSE",f"{metrics['RMSE']:.2e}")
st.divider()

# ── TABS ─────────────────────────────────────────────────────────────────────
(tab_formula,tab_walkthrough,tab_explorer,tab_data,
 tab_sens,tab_bench,tab_compare,tab_refs) = st.tabs([
    "📐 Formula Tuner","🗺️ PASTA Walkthrough","🔍 Threat Explorer",
    "📊 Dataset & Validation","📈 Sensitivity (OFAT)",
    "⚡ Scalability Benchmarks","🔬 Scenario Comparison","📚 References"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — FORMULA TUNER
# ═══════════════════════════════════════════════════════════════════════════
with tab_formula:
    st.subheader("📐 Interactive Formula Tuner")
    st.markdown("Adjust parameters and watch the **effort estimate update instantly** "
                "with a full step-by-step breakdown.")
    st.markdown(r"$$\hat{E}=\frac{(A_c \times A_v \times T \times T_w \times V \times \text{CVSS} \times C \times R)^S}{M \times Au}$$")
    st.divider()

    st.markdown("#### 🎚️ Single-Point Parameter Tuner")
    c1,c2,c3 = st.columns(3)
    with c1:
        ft_A  = st.slider("Asset Count (A)",  10,   5000, 500, key="ft_A")
        ft_Av = st.slider("Asset Value (Av)",  1,    100,  60, key="ft_Av")
        ft_T  = st.slider("Threat Vectors (T)",1,   2000, 100, key="ft_T")
    with c2:
        ft_V  = st.slider("Vulnerabilities",   1,  10000, 300, key="ft_V")
        ft_CV = st.slider("CVSS Score",       0.0,   10.0, 7.5, key="ft_CV",step=0.1)
        ft_C  = st.slider("Complexity (C)",    1,     10,   4,  key="ft_C")
    with c3:
        ft_R  = st.slider("Change Rate (R)",   1,     10,   4,  key="ft_R")
        ft_M  = st.slider("Org Maturity (M)",  1,     10,   5,  key="ft_M")
        ft_Au = st.slider("Automation (Au)",   1,     10,   5,  key="ft_Au")
        ft_S  = st.slider("Scaling Factor (S)",0.1,   3.0,  1.1,key="ft_S",step=0.1)
    ft_Tw = st.session_state["T_weight"]

    eA  = ft_A * ft_Av
    eV  = ft_V * max(ft_CV, 0.1)
    num = eA * ft_T * ft_Tw * eV * ft_C * ft_R
    den = max(ft_M * ft_Au, 1)
    effort = (float(num)**ft_S) / den

    st.divider()
    st.markdown("#### 🔢 Step-by-Step Calculation")
    bc1,bc2 = st.columns(2)
    with bc1:
        st.markdown(f"""
<div class="formula-box"><b>Step 1 — Effective Assets</b><br>
A_eff = {ft_A:,} × {ft_Av} = <b>{eA:,.0f}</b></div>
<div class="formula-box"><b>Step 2 — Effective Vulnerabilities</b><br>
V_eff = {ft_V:,} × {max(ft_CV,0.1):.1f} = <b>{eV:,.1f}</b></div>
<div class="formula-box"><b>Step 3 — Numerator Product</b><br>
{eA:,.0f} × {ft_T} × {ft_Tw:.2f} × {eV:,.1f} × {ft_C} × {ft_R} = <b>{num:.3e}</b></div>
""",unsafe_allow_html=True)
    with bc2:
        st.markdown(f"""
<div class="formula-box"><b>Step 4 — Denominator</b><br>
M × Au = {ft_M} × {ft_Au} = <b>{den}</b></div>
<div class="formula-box"><b>Step 5 — Apply Scaling Exponent S={ft_S}</b><br>
{num:.3e}^{ft_S} = <b>{num**ft_S:.3e}</b></div>
<div class="formula-box"><b>📌 Final Effort Estimate</b><br>
Ê = {num**ft_S:.3e} / {den} = <b style="color:#3b9dd2;font-size:1.15rem;">{effort:.4e}</b></div>
""",unsafe_allow_html=True)

    # Tornado chart
    st.divider()
    st.markdown("#### 🌪️ Tornado Chart — ±20% Parameter Sensitivity")
    st.caption("How much does Ê change when each variable moves ±20%?")
    pvals = {"AssetsCount":ft_A,"AssetValue":ft_Av,"ThreatVectors":ft_T,
             "Vulnerabilities":ft_V,"CVSSScore":ft_CV,"Complexity":ft_C,
             "ChangeRate":ft_R,"OrgMaturity":ft_M,"Automation":ft_Au}

    def calc_e(p,S,Tw):
        A=p["AssetsCount"]*p["AssetValue"]; V=p["Vulnerabilities"]*max(p["CVSSScore"],0.1)
        n_=A*p["ThreatVectors"]*Tw*V*p["Complexity"]*p["ChangeRate"]
        return (float(n_)**S)/max(p["OrgMaturity"]*p["Automation"],1)

    base_e = calc_e(pvals,ft_S,ft_Tw)
    trows=[]
    for p in pvals:
        el = calc_e({**pvals,p:pvals[p]*0.8},ft_S,ft_Tw)
        eh = calc_e({**pvals,p:pvals[p]*1.2},ft_S,ft_Tw)
        trows.append({"Variable":p,"Low":(el-base_e)/max(base_e,1e-12)*100,
                      "High":(eh-base_e)/max(base_e,1e-12)*100})
    td = pd.DataFrame(trows).sort_values("High",ascending=True)
    fig_t = go.Figure()
    fig_t.add_trace(go.Bar(y=td["Variable"],x=td["Low"],orientation="h",name="−20%",marker_color="#c0392b"))
    fig_t.add_trace(go.Bar(y=td["Variable"],x=td["High"],orientation="h",name="+20%",marker_color="#27ae60"))
    fig_t.update_layout(barmode="overlay",title="% Change in Ê (±20% perturbation per variable)",
        xaxis_title="% Change in Effort",height=360,margin=dict(l=0,r=0,t=40,b=0),
        legend=dict(orientation="h",y=1.08))
    st.plotly_chart(fig_t,use_container_width=True)

    # Live parameter influence line
    st.divider()
    st.markdown("#### 📊 Live Influence Line — Sweep Any Single Parameter")
    sweep_p = st.selectbox("Parameter to sweep:",list(pvals.keys()),key="ft_sweep")
    sp_lo,sp_hi = {"AssetsCount":(10,5000),"AssetValue":(1,100),"ThreatVectors":(1,2000),
        "Vulnerabilities":(1,10000),"CVSSScore":(0,10),"Complexity":(1,10),
        "ChangeRate":(1,10),"OrgMaturity":(1,10),"Automation":(1,10)}[sweep_p]
    xs_sw = np.linspace(sp_lo,sp_hi,100)
    ys_sw = [calc_e({**pvals,sweep_p:float(x)},ft_S,ft_Tw) for x in xs_sw]
    if log_mode: ys_sw = safe_log10(np.array(ys_sw))
    fig_inf = px.line(x=xs_sw,y=ys_sw,
        labels={"x":sweep_p,"y":"Ê"+((" (log₁₀)") if log_mode else "")},
        title=f"Ê as {sweep_p} varies (all others fixed at tuner values)",
        color_discrete_sequence=["#2e75b6"])
    fig_inf.add_vline(x=pvals[sweep_p],line_dash="dot",annotation_text="Current",
        annotation_position="top right")
    fig_inf.update_layout(height=300,margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_inf,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — PASTA WALKTHROUGH
# ═══════════════════════════════════════════════════════════════════════════
with tab_walkthrough:
    st.subheader("🗺️ PASTA 7-Stage Interactive Walkthrough")
    st.caption("Click any stage to explore its description, model variables, scalability bottleneck, and a live chart.")

    nav = st.columns(7)
    for i,(sn,si) in enumerate(PASTA_STAGES.items()):
        if nav[i].button(f"{si['icon']} {sn}",help=si["name"],use_container_width=True,
                         type="primary" if st.session_state["walkthrough_stage"]==sn else "secondary"):
            st.session_state["walkthrough_stage"]=sn
    st.divider()

    s   = st.session_state["walkthrough_stage"]
    inf = PASTA_STAGES[s]
    st.markdown(
        f"<div style='background:{inf['color']};padding:16px 20px;border-radius:10px;"
        f"color:white;margin-bottom:12px;'><h2 style='margin:0;color:white;'>"
        f"{inf['icon']} Stage {s}: {inf['name']}</h2>"
        f"<p style='margin:6px 0 0;opacity:0.9;'>{inf['description']}</p></div>",
        unsafe_allow_html=True)

    wc1,wc2 = st.columns([1.1,1])
    with wc1:
        st.markdown("##### 🎛️ Model Variables")
        for v in inf["model_vars"]:
            st.markdown(f"<span class='stage-pill' style='background:{inf['color']};'>{v}</span>",
                        unsafe_allow_html=True)
        st.markdown("##### ⚠️ Scalability Concern")
        st.markdown(f"<div class='callout'>🔴 <b>{inf['scalability_concern']}</b><br>"
                    f"Class: <code>{inf['bottleneck_class']}</code></div>",unsafe_allow_html=True)
        st.markdown("##### 🔬 Research Note")
        st.markdown(f"<div class='callout' style='border-color:#8e44ad;background:#f5eef8;'>"
                    f"{inf['research_note']}</div>",unsafe_allow_html=True)
    with wc2:
        st.markdown("##### 📈 Live Effort Sweep")
        sw_map={1:("OrgMaturity",(1,10)),2:("AssetsCount",asset_range),
                3:("Complexity",(1,10)),4:("ThreatVectors",threat_range),
                5:("Vulnerabilities",vuln_range),6:("ThreatVectors",threat_range),
                7:("Automation",(1,10))}
        sw_var,sw_bnds = sw_map[s]
        sw_xs = np.linspace(sw_bnds[0],sw_bnds[1],80)
        med   = {c:float(np.median(df[c])) for c in df.columns if c not in ["PredictedEffort","ActualEffort"]}
        sw_rows=[{**med,sw_var:float(v)} for v in sw_xs]
        sw_df  = pd.DataFrame(sw_rows)
        sw_df["Effort"] = model_predicted(sw_df,scaling_factor,st.session_state["T_weight"])
        y_sw = safe_log10(sw_df["Effort"]) if log_mode else sw_df["Effort"]
        fig_sw=px.line(x=sw_xs,y=y_sw,labels={"x":sw_var,"y":"Predicted Effort"},
            title=f"Effect of {sw_var} (others at dataset median)",
            color_discrete_sequence=[inf["color"]])
        fig_sw.update_layout(height=300,margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_sw,use_container_width=True)

    st.divider()
    st.markdown("##### 📋 All-Stage Scalability Overview")
    st.dataframe(pd.DataFrame([{
        "Stage":f"{si['icon']} {sn}. {si['name']}",
        "Complexity Class":si["bottleneck_class"],
        "Primary Variables":", ".join(si["model_vars"]),
        "Key Concern":si["scalability_concern"]}
        for sn,si in PASTA_STAGES.items()]),
        use_container_width=True,hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — THREAT EXPLORER
# ═══════════════════════════════════════════════════════════════════════════
with tab_explorer:
    st.subheader("🔍 Interactive Threat Vector Explorer")
    st.caption("Explore MITRE ATT&CK, ENISA, and VERIS threat selections and see how they affect T_weight and effort.")
    ex_std = st.radio("Framework:",["MITRE ATT&CK","ENISA Threat Landscape","VERIS"],horizontal=True)

    if ex_std=="MITRE ATT&CK":
        ex_sel = st.multiselect("Select MITRE techniques:",list(MITRE_TECHNIQUES.keys()),
            default=list(MITRE_TECHNIQUES.keys())[:5],key="ex_mitre")
        if ex_sel:
            ex_df = pd.DataFrame([{"Technique":t,"Tactic":MITRE_TECHNIQUES[t]["tactic"],
                "Severity":MITRE_TECHNIQUES[t]["severity"],"Weight":MITRE_TECHNIQUES[t]["weight"]}
                for t in ex_sel]).sort_values("Weight",ascending=False)
            ec1,ec2 = st.columns([1.3,1])
            with ec1:
                fig_b=px.bar(ex_df,x="Weight",y="Technique",orientation="h",color="Tactic",
                    title="Selected Technique Weights",color_discrete_sequence=px.colors.qualitative.Set2)
                fig_b.update_layout(height=300,margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig_b,use_container_width=True)
            with ec2:
                etw=sum(MITRE_TECHNIQUES[t]["weight"] for t in ex_sel)
                st.metric("Combined T_weight",f"{etw:.2f}")
                st.metric("# Techniques",len(ex_sel))
                st.metric("Dominant Tactic",ex_df.groupby("Tactic")["Weight"].sum().idxmax())
                td=ex_df.groupby("Tactic")["Weight"].sum().reset_index()
                fig_d=px.pie(td,values="Weight",names="Tactic",hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Set2)
                fig_d.update_layout(height=230,margin=dict(l=0,r=0,t=10,b=0),showlegend=False)
                st.plotly_chart(fig_d,use_container_width=True)
            # Radar
            if len(ex_sel)>=3:
                st.markdown("##### 🕸️ Technique Weight Radar")
                fig_r=go.Figure(go.Scatterpolar(
                    r=ex_df["Weight"].tolist()+[ex_df["Weight"].iloc[0]],
                    theta=ex_df["Technique"].tolist()+[ex_df["Technique"].iloc[0]],
                    fill="toself",line_color="#2e75b6"))
                fig_r.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,2])),
                    showlegend=False,height=360,margin=dict(l=40,r=40,t=30,b=20))
                st.plotly_chart(fig_r,use_container_width=True)
            # Impact on effort
            tw_exp = etw if etw>0 else 1.0
            e_sid  = df["PredictedEffort"].median()
            e_exp  = model_predicted(df,scaling_factor,T_weight=tw_exp).median()
            d_pct  = (e_exp-e_sid)/max(e_sid,1e-12)*100
            st.metric("Median Effort (Explorer Selection)",f"{e_exp:.2e}",
                delta=f"{d_pct:+.1f}% vs sidebar T_weight ({st.session_state['T_weight']:.2f})")

    elif ex_std=="ENISA Threat Landscape":
        ex_en=st.multiselect("Select ENISA threats:",list(ENISA_THREATS.keys()),
            default=list(ENISA_THREATS.keys())[:4],key="ex_enisa")
        if ex_en:
            en_df=pd.DataFrame([{"Threat":t,"Category":ENISA_THREATS[t]["category"],
                "Weight":ENISA_THREATS[t]["weight"],"Trend":ENISA_THREATS[t]["trend"]}
                for t in ex_en])
            ec1,ec2=st.columns(2)
            with ec1:
                fig_en=px.bar(en_df,x="Threat",y="Weight",color="Category",
                    title="ENISA Threat Weights",color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_en.update_layout(height=300,margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig_en,use_container_width=True)
            with ec2:
                st.dataframe(en_df.sort_values("Weight",ascending=False),
                    use_container_width=True,hide_index=True)
            etw_en=sum(ENISA_THREATS[t]["weight"] for t in ex_en)
            st.metric("Combined ENISA T_weight",f"{etw_en:.2f}")

    else:
        ex_ve=st.multiselect("Select VERIS actions:",list(VERIS_ACTIONS.keys()),
            default=list(VERIS_ACTIONS.keys())[:3],key="ex_veris")
        if ex_ve:
            ve_df=pd.DataFrame([{"Action":a,"Actor":VERIS_ACTIONS[a]["actor"],
                "Weight":VERIS_ACTIONS[a]["weight"]} for a in ex_ve])
            fig_ve=px.bar(ve_df,x="Action",y="Weight",color="Actor",title="VERIS Action Weights",
                color_discrete_map={"External":"#2980b9","Internal":"#e74c3c"})
            fig_ve.update_layout(height=300,margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_ve,use_container_width=True)
            st.metric("Combined VERIS T_weight",f"{sum(VERIS_ACTIONS[a]['weight'] for a in ex_ve):.2f}")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — DATASET & VALIDATION
# ═══════════════════════════════════════════════════════════════════════════
with tab_data:
    st.subheader("📊 Dataset & Model Validation")
    mc=st.columns(5)
    for i,(lbl,fmt) in enumerate(zip(["R²","R² log₁₀","MAE","RMSE","MAPE (%)"],
                                     [".4f",".4f",".2e",".2e",".2f"])):
        mc[i].metric(lbl,f"{metrics[lbl]:{fmt}}")
    st.caption("Synthetic targets → R² near 1 is expected. Use MAE/RMSE/MAPE to assess noise level.")

    dc1,dc2=st.columns(2)
    yt=safe_log10(df["ActualEffort"]) if log_mode else df["ActualEffort"]
    yp=safe_log10(df["PredictedEffort"]) if log_mode else df["PredictedEffort"]
    with dc1:
        fig_avp=px.scatter(x=yt,y=yp,opacity=0.3,
            labels={"x":"Actual"+(" (log₁₀)" if log_mode else ""),"y":"Predicted"+(" (log₁₀)" if log_mode else "")},
            title="Actual vs Predicted (hover to inspect)",color_discrete_sequence=["#2e75b6"])
        lm=[float(min(yt.min(),yp.min())),float(max(yt.max(),yp.max()))]
        fig_avp.add_trace(go.Scatter(x=lm,y=lm,mode="lines",line=dict(color="red",dash="dash"),name="Ideal"))
        fig_avp.update_layout(height=360,margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_avp,use_container_width=True)
    with dc2:
        resid=df["ActualEffort"].values-df["PredictedEffort"].values
        if log_mode: resid=safe_log10(df["ActualEffort"])-safe_log10(df["PredictedEffort"])
        fig_res=px.scatter(x=yt,y=resid,opacity=0.3,
            labels={"x":"Actual","y":"Residual"},title="Residuals vs Actual",
            color_discrete_sequence=["#c0392b"])
        fig_res.add_hline(y=0,line_dash="dash",line_color="black")
        fig_res.update_layout(height=360,margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_res,use_container_width=True)

    dist_d=safe_log10(df["PredictedEffort"]) if log_mode else df["PredictedEffort"]
    fig_dist=px.histogram(dist_d,nbins=60,
        labels={"value":"Predicted Effort"+(" (log₁₀)" if log_mode else "")},
        title="Distribution of Predicted Effort",color_discrete_sequence=["#1abc9c"])
    fig_dist.update_layout(height=260,margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_dist,use_container_width=True)

    # Correlation heatmap
    st.markdown("#### 🔗 Feature Correlation Heatmap")
    corr=df[["AssetsCount","AssetValue","ThreatVectors","Vulnerabilities","CVSSScore",
             "Complexity","ChangeRate","OrgMaturity","Automation","PredictedEffort"]].corr()
    fig_h=px.imshow(corr,text_auto=".2f",aspect="auto",color_continuous_scale="RdBu_r",
        zmin=-1,zmax=1,title="Pearson Correlation Matrix")
    fig_h.update_layout(height=420,margin=dict(l=0,r=0,t=50,b=0))
    st.plotly_chart(fig_h,use_container_width=True)

    st.dataframe(df.head(20),use_container_width=True)
    dl1,dl2=st.columns(2)
    with dl1:
        st.download_button("📥 Dataset (CSV)",df.to_csv(index=False).encode(),
                           "pasta_dataset.csv","text/csv")
    with dl2:
        cfg={"num_samples":num_samples,"S":scaling_factor,
             "ranges":{k:list(v) for k,v in ranges.items()},
             "noise":{"type":noise_type,"pct":noise_pct},"T_weight":st.session_state["T_weight"]}
        st.download_button("🧾 Config (JSON)",json.dumps(cfg,indent=2).encode(),
                           "config.json","application/json")
    st.markdown("#### 📐 Descriptive Statistics")
    st.dataframe(df.describe().round(3),use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — SENSITIVITY (OFAT)
# ═══════════════════════════════════════════════════════════════════════════
with tab_sens:
    st.subheader("📈 One-Factor-At-A-Time (OFAT) Sensitivity Analysis")
    st.caption("Sweep one variable across its full range; all others held at dataset medians. Updates live.")

    meds={c:float(np.median(df[c])) for c in
          ["AssetsCount","AssetValue","ThreatVectors","Vulnerabilities",
           "CVSSScore","Complexity","ChangeRate","OrgMaturity","Automation"]}
    int_v={"AssetsCount","ThreatVectors","Vulnerabilities","Complexity","ChangeRate","OrgMaturity","Automation"}
    bnds={**{k:ranges[k] for k in int_v},
          "AssetValue":(1,100),"CVSSScore":(0.0,10.0)}
    lo,hi=bnds[selected_var]
    xs=np.linspace(lo,hi,ofat_points)
    orows=[{**meds,selected_var:float(int(round(v))) if selected_var in int_v else float(v)} for v in xs]
    od=pd.DataFrame(orows)
    od["PredictedEffort"]=model_predicted(od,scaling_factor,st.session_state["T_weight"])
    yo=safe_log10(od["PredictedEffort"]) if log_mode else od["PredictedEffort"]

    fig_of=px.line(x=xs,y=yo,
        labels={"x":selected_var,"y":"Predicted Effort"+(" (log₁₀)" if log_mode else "")},
        title=f"OFAT: Effect of '{selected_var}'",color_discrete_sequence=["#8e44ad"])
    fig_of.add_vline(x=meds[selected_var],line_dash="dot",
        annotation_text="Median",annotation_position="top right")
    fig_of.update_layout(height=380,margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_of,use_container_width=True)

    # All-variable normalised
    st.markdown("#### 📊 Normalised OFAT — All Variables Overlaid")
    fig_all=go.Figure()
    cols_a=px.colors.qualitative.Set1
    for i,var in enumerate(list(bnds.keys())):
        bl,bh=bnds[var]; xs_n=np.linspace(bl,bh,ofat_points)
        nx=(xs_n-bl)/max(bh-bl,1e-12)
        rn=[{**meds,var:float(int(round(v))) if var in int_v else float(v)} for v in xs_n]
        tn=pd.DataFrame(rn); te=model_predicted(tn,scaling_factor,st.session_state["T_weight"])
        tl=safe_log10(te) if log_mode else te
        emin,emax=float(tl.min()),float(tl.max())
        ne=(tl-emin)/max(emax-emin,1e-12)
        fig_all.add_trace(go.Scatter(x=nx,y=ne,mode="lines",name=var,
            line=dict(color=cols_a[i%len(cols_a)],width=1.8)))
    fig_all.update_layout(title="Normalised Sensitivity (all variables, 0→1 scale)",
        xaxis_title="Normalised Variable Range",yaxis_title="Normalised Effort",
        height=400,margin=dict(l=0,r=0,t=40,b=0),
        legend=dict(orientation="h",y=1.06))
    st.plotly_chart(fig_all,use_container_width=True)

    # Elasticities
    st.markdown("#### 📐 Elasticity Table")
    cf={v:scaling_factor for v in ["AssetsCount","AssetValue","ThreatVectors",
        "Vulnerabilities","CVSSScore","Complexity","ChangeRate"]}
    cf.update({"OrgMaturity":-1.0,"Automation":-1.0})

    def emp_e(var,base,d=0.05):
        x0=max(1e-9,float(base[var]))
        step=max(1.0,x0*d) if var in int_v else (x0*d or d)
        x1=x0+step; r0,r1=base.copy(),base.copy(); r1[var]=x1
        E0=float(model_predicted(pd.DataFrame([r0]),scaling_factor).iloc[0])
        E1=float(model_predicted(pd.DataFrame([r1]),scaling_factor).iloc[0])
        dE=(E1-E0)/max(E0,1e-12); dX=(x1-x0)/max(x0,1e-12)
        return dE/dX if dX else np.nan

    emp={v:emp_e(v,meds) for v in cf}
    ed=pd.DataFrame({"Variable":list(cf.keys()),
        "Closed-Form ε":[round(cf[v],3) for v in cf],
        "Empirical ε":[round(emp[v],3) for v in cf],
        "Role":["Numerator" if cf[v]>0 else "Denominator" for v in cf]})
    st.dataframe(ed,use_container_width=True,hide_index=True)
    fig_el=px.bar(ed,x="Variable",y="Empirical ε",color="Role",
        color_discrete_map={"Numerator":"#2e75b6","Denominator":"#c0392b"},
        title="Empirical Elasticities at Median Operating Point")
    fig_el.add_hline(y=0,line_dash="dash",line_color="black")
    fig_el.update_layout(height=300,margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_el,use_container_width=True)
    st.download_button("📥 OFAT Data (CSV)",od.to_csv(index=False).encode(),
                       f"ofat_{selected_var}.csv","text/csv")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — SCALABILITY BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════
with tab_bench:
    st.subheader("⚡ Quantitative Scalability Benchmarks")
    st.markdown("<div class='callout'><b>Research hypothesis:</b> Wall-clock time and memory should "
                "scale predictably with N. Sub-linear = well-optimised; slope > 1.5 = bottleneck.</div>",
                unsafe_allow_html=True)

    bc1,bc2,bc3=st.columns(3)
    with bc1: bench_n=st.slider("N evaluation points",5,20,10)
    with bc2: bench_max=st.number_input("Max N",value=10_000,step=1_000,min_value=500,max_value=50_000)
    with bc3: bench_scale=st.radio("N spacing",["Linear","Log"],horizontal=True)

    bench_sizes=tuple(int(x) for x in (
        np.geomspace(100,bench_max,bench_n) if bench_scale=="Log"
        else np.linspace(100,bench_max,bench_n,dtype=int)))

    if st.button("▶ Run Scalability Benchmark",type="primary"):
        with st.spinner(f"Benchmarking {len(bench_sizes)} problem sizes…"):
            bd=run_scalability_benchmark(bench_sizes,ranges,rng_seed,scaling_factor,
                st.session_state["T_weight"],use_asset_value,asset_value,
                use_cvss_baseline,cvss_baseline)
        st.session_state["benchmark_results"]=bd

    if st.session_state["benchmark_results"] is not None:
        bd=st.session_state["benchmark_results"]

        st.markdown("#### 📈 Scalability Profiles (hover, zoom, pan)")
        fig_bm=make_subplots(rows=2,cols=2,
            subplot_titles=("Total Wall-Clock Time","Time Breakdown by Stage",
                           "Peak Memory Usage","Throughput (samples/s)"))
        fig_bm.add_trace(go.Scatter(x=bd["N"],y=bd["Total (s)"],mode="lines+markers",
            name="Total Time",line=dict(color="#2e75b6")),row=1,col=1)
        fig_bm.add_trace(go.Scatter(x=bd["N"],y=bd["Gen Time (s)"],mode="lines",
            name="Data Gen",fill="tozeroy",line=dict(color="#1abc9c")),row=1,col=2)
        fig_bm.add_trace(go.Scatter(x=bd["N"],y=bd["Pred Time (s)"],mode="lines",
            name="Prediction",fill="tonexty",line=dict(color="#e67e22")),row=1,col=2)
        fig_bm.add_trace(go.Scatter(x=bd["N"],y=bd["Gen Mem (KB)"],mode="lines+markers",
            name="Gen Mem",line=dict(color="#8e44ad"),marker=dict(symbol="square")),row=2,col=1)
        fig_bm.add_trace(go.Scatter(x=bd["N"],y=bd["Pred Mem (KB)"],mode="lines+markers",
            name="Pred Mem",line=dict(color="#c0392b"),marker=dict(symbol="triangle-up")),row=2,col=1)
        fig_bm.add_trace(go.Bar(x=bd["N"],y=bd["Throughput"],name="Throughput",
            marker_color="#2980b9"),row=2,col=2)
        fig_bm.update_xaxes(title_text="N (Samples)")
        fig_bm.update_yaxes(title_text="Seconds",row=1,col=1)
        fig_bm.update_yaxes(title_text="Seconds",row=1,col=2)
        fig_bm.update_yaxes(title_text="KB",row=2,col=1)
        fig_bm.update_yaxes(title_text="Samples/s",row=2,col=2)
        fig_bm.update_layout(height=540,showlegend=True,margin=dict(l=0,r=0,t=60,b=0),
            legend=dict(orientation="h",y=-0.08))
        st.plotly_chart(fig_bm,use_container_width=True)

        # Log-log complexity
        st.markdown("#### 📐 Empirical Complexity Estimate (Log-Log Fit)")
        lN=np.log(bd["N"].values.astype(float))
        lT=np.log(bd["Total (s)"].values.astype(float)+1e-9)
        slope,intercept=np.polyfit(lN,lT,1)
        fit=np.exp(intercept+slope*lN)
        fig_ll=go.Figure()
        fig_ll.add_trace(go.Scatter(x=bd["N"],y=bd["Total (s)"],mode="markers",name="Observed",
            marker=dict(color="#2e75b6",size=8)))
        fig_ll.add_trace(go.Scatter(x=bd["N"],y=fit,mode="lines",name=f"Fit O(N^{slope:.2f})",
            line=dict(color="#e74c3c",dash="dash",width=2)))
        fig_ll.update_layout(title=f"Log-Log Fit: Empirical Complexity ≈ O(N^{slope:.3f})",
            xaxis=dict(title="N",type="log"),yaxis=dict(title="Total Time (s)",type="log"),
            height=320,margin=dict(l=0,r=0,t=50,b=0))
        st.plotly_chart(fig_ll,use_container_width=True)

        if slope<1.1:   tag,msg="🟢 Near-Linear","Scales efficiently. Suitable for large deployments."
        elif slope<1.5: tag,msg="🟡 Mildly Super-Linear","Some overhead. Acceptable for medium-scale systems."
        else:           tag,msg="🔴 Super-Linear","Significant bottleneck. Optimisation recommended."
        st.markdown(f"<div class='callout'><b>{tag}</b> — {msg} "
                    f"Fitted k = <b>{slope:.3f}</b></div>",unsafe_allow_html=True)

        st.markdown("#### 📋 Raw Benchmark Table")
        st.dataframe(bd.style.format({"Gen Time (s)":"{:.5f}","Pred Time (s)":"{:.5f}",
            "Total (s)":"{:.5f}","Gen Mem (KB)":"{:.1f}","Pred Mem (KB)":"{:.1f}",
            "Throughput":"{:,.0f}"}),use_container_width=True)
        st.download_button("📥 Benchmark Data (CSV)",bd.to_csv(index=False).encode(),
                           "pasta_benchmark.csv","text/csv")
    else:
        st.info("👆 Click **Run Scalability Benchmark** to generate empirical measurements.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 7 — SCENARIO COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("🔬 Scenario Comparison")
    st.caption("Define two configurations (e.g. small vs large deployment) and compare side-by-side.")

    def sc_inputs(label,kp,defs):
        with st.expander(f"⚙️ {label}",expanded=True):
            c1,c2,c3=st.columns(3)
            with c1:
                a =st.slider("Asset Count", 10,10000,defs["A"],  key=f"{kp}_A")
                av=st.slider("Asset Value",  1,  100,defs["Av"], key=f"{kp}_Av")
                t =st.slider("Threat Vecs",  1, 5000,defs["T"],  key=f"{kp}_T")
            with c2:
                v =st.slider("Vulnerabilities",1,20000,defs["V"],key=f"{kp}_V")
                cv=st.slider("CVSS Score",0.0,10.0,defs["CV"],   key=f"{kp}_CV",step=0.1)
                c =st.slider("Complexity",  1,   10,defs["C"],   key=f"{kp}_C")
            with c3:
                r =st.slider("Change Rate", 1,   10,defs["R"],   key=f"{kp}_R")
                m =st.slider("Org Maturity",1,   10,defs["M"],   key=f"{kp}_M")
                au=st.slider("Automation",  1,   10,defs["Au"],  key=f"{kp}_Au")
                s =st.slider("Scaling (S)",0.1, 3.0,defs["S"],   key=f"{kp}_S",step=0.1)
        return {"A":a,"Av":av,"T":t,"V":v,"CV":cv,"C":c,"R":r,"M":m,"Au":au,"S":s}

    sca,scb=st.columns(2)
    with sca: pA=sc_inputs("Scenario A — Baseline","scA",
        {"A":200,"Av":50,"T":50,"V":200,"CV":6.0,"C":3,"R":3,"M":6,"Au":6,"S":1.1})
    with scb: pB=sc_inputs("Scenario B — Scaled-Up","scB",
        {"A":2000,"Av":70,"T":500,"V":2000,"CV":8.0,"C":7,"R":7,"M":3,"Au":3,"S":1.3})

    def ev(p,Tw):
        A=p["A"]*p["Av"]; V=p["V"]*max(p["CV"],0.1)
        return (float(A*p["T"]*Tw*V*p["C"]*p["R"])**p["S"])/max(p["M"]*p["Au"],1)

    tw=st.session_state["T_weight"]
    eA=ev(pA,tw); eB=ev(pB,tw); ratio=eB/max(eA,1e-12)

    st.divider()
    st.markdown("### 📊 Comparison Results")
    cm1,cm2,cm3=st.columns(3)
    cm1.metric("Scenario A Effort",f"{eA:.3e}")
    cm2.metric("Scenario B Effort",f"{eB:.3e}",delta=f"{(ratio-1)*100:+.1f}% vs A")
    cm3.metric("B / A Ratio",f"{ratio:.2f}×")

    pkeys=["A","Av","T","V","CV","C","R","M","Au","S"]
    plbls=["AssetsCount","AssetValue","ThreatVectors","Vulnerabilities",
           "CVSSScore","Complexity","ChangeRate","OrgMaturity","Automation","S"]
    cmp=[{"Parameter":l,"Scenario A":pA[k],"Scenario B":pB[k]} for k,l in zip(pkeys,plbls)]
    cdf=pd.DataFrame(cmp)
    cmp1,cmp2=st.columns(2)
    with cmp1:
        fig_cmp=px.bar(cdf.melt(id_vars="Parameter",var_name="Scenario",value_name="Value"),
            x="Parameter",y="Value",color="Scenario",barmode="group",
            color_discrete_map={"Scenario A":"#2e75b6","Scenario B":"#e74c3c"},
            title="Parameter Values: A vs B")
        fig_cmp.update_xaxes(tickangle=35)
        fig_cmp.update_layout(height=340,margin=dict(l=0,r=0,t=40,b=60))
        st.plotly_chart(fig_cmp,use_container_width=True)
    with cmp2:
        def norm_p(p):
            maxvals={"A":10000,"Av":100,"T":5000,"V":20000,"CV":10,"C":10,"R":10,"M":10,"Au":10,"S":3}
            return [(p[k]/maxvals[k]) for k in pkeys]
        fig_rad=go.Figure()
        for vals,name,col in [(norm_p(pA),"Scenario A","#2e75b6"),(norm_p(pB),"Scenario B","#e74c3c")]:
            fig_rad.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=plbls+[plbls[0]],
                fill="toself",name=name,line_color=col))
        fig_rad.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])),
            title="Normalised Parameter Radar",height=340,margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig_rad,use_container_width=True)

    # Log contribution breakdown
    st.markdown("#### 🔢 Log₁₀ Contribution of Each Term")
    st.caption("Shows which term dominates the effort score in each scenario.")
    def log_c(p,Tw):
        return {"A_count":np.log10(max(p["A"],1)),"A_val":np.log10(max(p["Av"],1)),
                "T":np.log10(max(p["T"],1)),"T_weight":np.log10(max(Tw,0.001)),
                "V":np.log10(max(p["V"],1)),"CVSS":np.log10(max(p["CV"],0.1)),
                "Complexity":np.log10(max(p["C"],1)),"ChangeRate":np.log10(max(p["R"],1)),
                "OrgMaturity":-np.log10(max(p["M"],1)),"Automation":-np.log10(max(p["Au"],1))}
    la=log_c(pA,tw); lb=log_c(pB,tw)
    ldf=pd.DataFrame({"Term":list(la.keys()),"Scenario A":list(la.values()),"Scenario B":list(lb.values())})
    fig_lc=px.bar(ldf.melt(id_vars="Term",var_name="Scenario",value_name="log₁₀ Contribution"),
        x="Term",y="log₁₀ Contribution",color="Scenario",barmode="group",
        color_discrete_map={"Scenario A":"#2e75b6","Scenario B":"#e74c3c"},
        title="log₁₀ Contribution per Term (negative = denominator)")
    fig_lc.add_hline(y=0,line_dash="dash")
    fig_lc.update_xaxes(tickangle=25)
    fig_lc.update_layout(height=320,margin=dict(l=0,r=0,t=40,b=60))
    st.plotly_chart(fig_lc,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 8 — REFERENCES
# ═══════════════════════════════════════════════════════════════════════════
with tab_refs:
    st.subheader("📚 Academic References & Framework Sources")
    st.markdown("""
### Core PASTA Framework
1. UcedaVélez, T. & Morana, M.M. (2015). *Risk Centric Threat Modeling: Process for Attack
   Simulation and Threat Analysis*. Wiley. ISBN 978-0-470-50096-5.
2. Shostack, A. (2014). *Threat Modeling: Designing for Security*. Wiley.

### Threat Taxonomies
3. **MITRE ATT&CK** — https://attack.mitre.org
4. **ENISA Threat Landscape 2024** — https://www.enisa.europa.eu/topics/cyber-threats/enisa-threat-landscape
5. **VERIS Community** — http://veriscommunity.net

### Vulnerability Scoring
6. **NIST NVD / CVSS v3.1** — https://nvd.nist.gov/vuln-metrics/cvss
7. FIRST.org. (2019). *CVSS v3.1 Specification Document*.

### Scalability & ML-Assisted Threat Modeling
8. Xiong, W. & Lagerström, R. (2019). Threat modeling — A systematic literature review.
   *Computers & Security*, 84, 53–69.
9. Jha, S. et al. (2002). Two formal analyses of attack graphs. *IEEE CSFW*, 49–63.
   (NP-hardness of attack path enumeration)

### Model Variables Quick Reference
| Symbol | Variable | PASTA Stage | Elasticity |
|--------|----------|-------------|------------|
| A_count | Asset Count | 2 — Technical Scope | +S |
| A_val | Asset Value | 7 — Risk & Impact | +S |
| T | Threat Vectors | 4 — Threat Analysis | +S |
| T_w | MITRE T_weight | 4 | +1 (linear) |
| V | Vulnerability Count | 5 — Vulnerabilities | +S |
| CVSS | Severity Score | 5 | +S |
| C | Complexity | 3 — Decomposition | +S |
| R | Change Rate | 3 | +S |
| M | Org Maturity | 1 — Objectives | −1 |
| Au | Automation | 1, 7 | −1 |
| S | Scaling Exponent | — | Non-linear amplifier |
""")
    st.info("💡 **For your thesis**: Cite this tool as implementing the PASTA effort model with "
            "quantitative scalability profiling. All results are reproducible via the JSON config export.")
