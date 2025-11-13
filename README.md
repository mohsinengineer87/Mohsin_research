# PASTA Effort Estimator (Streamlit)

Interactive Streamlit app to experiment with a PASTA-based effort/scalability model:

\[\hat{E} = \frac{(A \times T \times V \times C \times R)^S}{M \times Au}\]

- **A**: Assets, **T**: ThreatVectors, **V**: Vulnerabilities, **C**: Complexity, **R**: ChangeRate  
- **M**: OrgMaturity, **Au**: Automation, **S**: Scaling factor

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)<br>
*(After you deploy, replace the link above with your app URL.)*

## ✨ Features
- **Noise controls**: None / Gaussian / Laplace, scaled as % of mean prediction.
- **Log mode**: Compute plots and R² in **log10 space** (robust to large scales).
- **Sensitivity analysis**: One-Factor-At-a-Time (OFAT) sweep + **elasticities**.
- **Reproducibility**: Independent seeds for data & noise.
- **Downloads**: Dataset (CSV), config (JSON), plots (PNG), elasticities (CSV).

## 🚀 Run locally
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy on Streamlit Community Cloud
1. Push this folder to a **public GitHub repo**.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, select your repo & branch.
4. Set **Main file path** to `app.py`.
5. Click **Deploy**.

> After deploy, update this README with your app URL and optionally add screenshots in `screenshots/`.

## 📂 File structure
```
.
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
├── .gitignore
└── LICENSE
```

## 🧾 License
MIT License — feel free to use and modify with attribution.
