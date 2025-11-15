import streamlit as st

# ===============================
# 💡 MITRE ATT&CK Matrix Data
# ===============================
mitre_attack_matrix = {
    "Initial Access": ["Phishing (T1566.001)", "Drive-by Compromise (T1189)", "Valid Accounts (T1078)"],
    "Execution": ["PowerShell (T1059.001)", "Command-Line Interface (T1059.003)", "Scripting (T1064)"],
    "Persistence": ["Registry Run Keys (T1547.001)", "Scheduled Task (T1053.005)", "Startup Folder (T1547.001)"],
    "Privilege Escalation": ["Bypass UAC (T1548.002)", "Token Impersonation (T1134.001)", "Exploitation for Privilege (T1068)"],
    "Defense Evasion": ["Obfuscated Files (T1027)", "Disable Security Tools (T1562.001)", "Clear Command History (T1070.003)"],
    "Exfiltration": ["Exfiltration Over Web (T1041)", "Automated Exfiltration (T1020)", "Exfil via Cloud (T1537)"],
}

# ===============================
# 📊 Visual MITRE ATT&CK Matrix
# ===============================
with st.expander('📊 MITRE ATT&CK Matrix (Visual Threat Vectors)'):
    for tactic, techniques in mitre_attack_matrix.items():
        st.markdown(f'### 🛡️ {tactic}')
        st.markdown("\n".join([f'- {tech}' for tech in techniques]))
