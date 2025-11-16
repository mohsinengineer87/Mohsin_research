
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Unified Threat-Driven Effort Model", layout="wide")

st.title("🔐 Threat-Driven Effort & Scalability Model")
st.markdown("This unified model uses selected threat vectors from MITRE, ENISA, and VERIS to compute effort. No asset sliders. Everything works together based on threat intelligence.")

# ==========================
# Unified Threat Vector List
# ==========================
all_threats = [
    # MITRE
    'Phishing (T1566.001)',
    'PowerShell (T1059.001)',
    'Registry Run Keys (T1547.001)',
    'Scheduled Task (T1053.005)',
    'Command-Line Interface (T1059.003)',
    'Token Impersonation (T1134.001)',
    'Bypass UAC (T1548.002)',
    'Exfiltration Over Web (T1041)',
    # ENISA
    'Ransomware',
    'Phishing',
    'Data Breaches',
    'Insider Threats',
    'Denial of Service',
    # VERIS
    'Malware',
    'Hacking',
    'Social',
    'Misuse',
    'Physical',
    'Error'
]

selected_threats = st.multiselect(
    "📌 Select Threat Vectors (MITRE + ENISA + VERIS)", 
    options=all_threats,
    default=[],
    key='unified_threat_selector'
)

# ==========================
# Threat Weight Mapping
# ==========================
threat_priority_weights = {
    'Phishing (T1566.001)': 1.5,
    'PowerShell (T1059.001)': 1.4,
    'Registry Run Keys (T1547.001)': 1.2,
    'Scheduled Task (T1053.005)': 1.1,
    'Command-Line Interface (T1059.003)': 1.3,
    'Token Impersonation (T1134.001)': 1.4,
    'Bypass UAC (T1548.002)': 1.6,
    'Exfiltration Over Web (T1041)': 1.7,
    'Ransomware': 1.8,
    'Phishing': 1.5,
    'Data Breaches': 1.7,
    'Insider Threats': 1.5,
    'Denial of Service': 1.2,
    'Malware': 1.4,
    'Hacking': 1.6,
    'Social': 1.3,
    'Misuse': 1.2,
    'Physical': 1.0,
    'Error': 1.1
}

# ==========================
# Effort Equation (Unified)
# ==========================
scaling_factor = st.slider("🔁 Scaling Factor (S)", 1.0, 5.0, 2.0, 0.1)
weighted_threat_score = sum(threat_priority_weights.get(t, 1.0) for t in selected_threats)

effort = weighted_threat_score ** scaling_factor
st.metric("🧠 Effort Score", round(effort, 2))

# ==========================
# Chart Display
# ==========================
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4, 5], [weighted_threat_score ** s for s in [1, 2, 3, 4, 5]], marker='o')
ax.set_title("📈 Effort Growth with Scaling Factor")
ax.set_xlabel("Scaling Factor (S)")
ax.set_ylabel("Effort")
st.pyplot(fig)

# ==========================
# Downloadable Chart
# ==========================
import io
buf = io.BytesIO()
fig.savefig(buf, format="png")
st.download_button(
    label="📥 Download Effort Chart",
    data=buf.getvalue(),
    file_name="effort_chart.png",
    mime="image/png"
)
