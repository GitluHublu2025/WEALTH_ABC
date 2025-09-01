
# Global Portfolio Dashboard — Clean Deployment Package

This package contains a single-file Streamlit app (`GlobalDashboard_clean.py`) and supporting files to deploy directly to Streamlit Cloud or run locally.

## Files
- `GlobalDashboard_clean.py` — final Streamlit app (single script).
- `requirements.txt` — Python dependencies.
- `india_template.csv` — sample India portfolio CSV template.
- `us_template.csv` — sample US portfolio CSV template.
- `README.md` — this file.

## How to run locally
1. Create a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run GlobalDashboard_clean.py
   ```

## Notes
- The app uses a *safe* Yahoo Finance downloader to avoid blocking Streamlit UI.
- India tickers should be uploaded **without** the `.NS` suffix (the app appends it automatically).
- US tickers use their regular tickers (AAPL, MSFT, ...).
- The app intentionally **removes** fundamentals (Trailing PE, Forward PE, EPS, Dividend Yield, Beta) per your request and focuses on P/L and planning modules.
- Use the sidebar to upload portfolios, choose the lookback period, enable/disable charts, and access estimation modules (Projection Settings, Bonds, Fixed Deposits, Target ROI Planner).
