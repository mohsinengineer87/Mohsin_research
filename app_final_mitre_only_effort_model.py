
import streamlit as st
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="MITRE-Driven Effort Model", layout="wide")

st.title("🛡️ MITRE-Based Threat Modeling and Effort Estimation")

# ------------------------
# MITRE ATT&CK Techniques
# ------------------------
mitre_techniques = [
    'Phishing (T1566.001)',
    'PowerShell (T1059.001)',
    'Registry Run Keys (T1547.001)',
    'Scheduled Task (T1053.005)',
    'Command-Line Interface (T1059.003)',
    'Token Impersonation (T1134.001)',
    'Bypass UAC (T1548.002)',
    'Exfiltration Over Web (T1041)',
]

# ------------------------
# Threat Priority Weights
# ------------------------
technique_weights = {
    'Phishing (T1566.001)': 1.5,
    'PowerShell (T1059.001)': 1.4,
    'Registry Run Keys (T1547.001)': 1.2,
    'Scheduled Task (T1053.005)': 1.1,
    'Command-Line Interface (T1059.003)': 1.3,
    'Token Impersonation (T1134.001)': 1.4,
    'Bypass UAC (T1548.002)': 1.6,
    'Exfiltration Over Web (T1041)': 1.7,
}

selected_techniques = st.multiselect(
    "🎯 Select MITRE Techniques:",
    options=mitre_techniques,
    default=[],
    key="mitre_select"
)

# ------------------------
# Compute Weighted Score
# ------------------------
weighted_threat_score = sum(technique_weights.get(t, 1.0) for t in selected_techniques)

# ------------------------
# Effort Equation
# ------------------------
scaling_factor = st.slider("⚙️ Scaling Factor (S)", 1.0, 5.0, 2.0, 0.1)
effort = weighted_threat_score ** scaling_factor

st.metric("📊 Computed Effort", round(effort, 2))

# ------------------------
# Effort Growth Chart
# ------------------------
fig, ax = plt.subplots()
x_vals = [1, 2, 3, 4, 5]
y_vals = [weighted_threat_score ** s for s in x_vals]
ax.plot(x_vals, y_vals, marker='o')
ax.set_title("Effort Growth vs Scaling Factor")
ax.set_xlabel("Scaling Factor (S)")
ax.set_ylabel("Effort")
st.pyplot(fig)

# ------------------------
# Download Chart
# ------------------------
buf = io.BytesIO()
fig.savefig(buf, format="png")
st.download_button(
    label="📥 Download Effort Chart",
    data=buf.getvalue(),
    file_name="effort_chart.png",
    mime="image/png"
)
