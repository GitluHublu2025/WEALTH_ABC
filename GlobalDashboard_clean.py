# -*- coding: utf-8 -*-
"""
GlobalDashboard_final.py
Final Streamlit app for Global Portfolio Dashboard with:
- India & US stocks (CSV upload)
- INR & USD Bonds / FDs (CSV upload + manual entry)
- Live P&L, currency conversion, benchmarks (1mo–2y)
- INR depreciation slider
- ROI Planner with allocations (INR, USD, %)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import altair as alt
from io import BytesIO
from datetime import date

st.set_page_config(page_title="Global Portfolio Dashboard — Final", layout="wide")
st.title("📈 Global Portfolio Dashboard — Final")

# ------------------ Sidebar: uploads & settings ------------------
st.sidebar.header("Uploads & Settings")

# Stock CSV uploads
india_file = st.sidebar.file_uploader("Upload Indian Portfolio CSV (Ticker,Quantity,AvgCost) — no .NS", type=["csv"])
us_file = st.sidebar.file_uploader("Upload US Portfolio CSV (Ticker,Quantity,AvgCost)", type=["csv"])

# Bonds & FDs CSV uploads
bonds_inr_file = st.sidebar.file_uploader("Upload Bonds (INR) CSV (Principal,Date,InterestRate%)", type=["csv"])
bonds_usd_file = st.sidebar.file_uploader("Upload Bonds (USD) CSV (Principal,Date,InterestRate%)", type=["csv"])
fds_inr_file = st.sidebar.file_uploader("Upload Fixed Deposits (INR) CSV (Principal,Date,InterestRate%)", type=["csv"])
fds_usd_file = st.sidebar.file_uploader("Upload Fixed Deposits (USD) CSV (Principal,Date,InterestRate%)", type=["csv"])

# Display currency and lookback
display_currency = st.sidebar.radio("Display currency", ["INR", "USD"], index=0)
lookback = st.sidebar.selectbox("Performance lookback period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
show_chart = st.sidebar.checkbox("Show Benchmarks & Portfolio Chart", value=True)

# Growth & depreciation sliders
st.sidebar.subheader("Projection Inputs")
india_growth = st.sidebar.slider("India stocks growth rate (%)", -50.0, 50.0, 12.0, 0.1)
us_growth = st.sidebar.slider("US stocks growth rate (%)", -50.0, 50.0, 8.0, 0.1)
target_roi = st.sidebar.slider("Target overall ROI (%)", -50.0, 50.0, 10.0, 0.1)
inr_depr = st.sidebar.slider("INR depreciation vs USD (% per year)", 0.0, 10.0, 3.0, 0.1)

# Auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh
    _ = st_autorefresh(interval=5 * 60 * 1000, key="autorefresh_final")
except Exception:
    pass

# ------------------ Utilities ------------------
@st.cache_data(ttl=300)
def safe_download(tickers, period="1y", interval="1d", group_by="ticker", threads=True):
    try:
        df = yf.download(tickers, period=period, interval=interval, progress=False,
                         group_by=group_by, threads=threads)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def batch_last_close(symbols):
    if not symbols: return {}
    out = {s: np.nan for s in symbols}
    data = safe_download(symbols, period="5d", interval="1d", group_by="ticker")
    def extract_close(d, sym):
        try:
            if isinstance(d, pd.DataFrame) and isinstance(d.columns, pd.MultiIndex):
                ser = d[sym]["Close"].dropna()
                if not ser.empty: return float(ser.iloc[-1])
            if "Close" in d.columns:
                ser = d["Close"].dropna()
                if not ser.empty: return float(ser.iloc[-1])
        except Exception:
            pass
        try:
            h = yf.Ticker(sym).history(period="5d", interval="1d")
            if not h.empty: return float(h["Close"].dropna().iloc[-1])
        except Exception:
            pass
        return np.nan
    if not data.empty:
        for s in symbols: out[s] = extract_close(data, s)
    else:
        for s in symbols: out[s] = extract_close(pd.DataFrame(), s)
    return out

@st.cache_data(ttl=600)
def get_fx_rate():
    try:
        finf = yf.Ticker("USDINR=X").fast_info
        if finf and finf.get("lastPrice"): return float(finf["lastPrice"])
    except Exception: pass
    d = safe_download("USDINR=X", period="1d", interval="5m")
    if not d.empty and "Close" in d.columns:
        return float(d["Close"].dropna().iloc[-1])
    return 83.0

fx_rate = get_fx_rate()

# ------------------ Loaders ------------------
def load_stock_csv(f, is_india=True):
    if f is None: return pd.DataFrame()
    df = pd.read_csv(f)
    df = df.rename(columns={"Ticker":"ticker","Quantity":"shares","AvgCost":"avg_cost"})
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["avg_cost"] = pd.to_numeric(df["avg_cost"], errors="coerce")
    if is_india:
        df["ticker"] = df["ticker"].astype(str).apply(lambda x: x if x.endswith(".NS") else f"{x}.NS")
        df["currency"] = "INR"
    else:
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["currency"] = "USD"
    return df

def load_asset_csv(f, currency="INR"):
    if f is None: return pd.DataFrame()
    df = pd.read_csv(f)
    df = df.rename(columns={c:c.strip().title() for c in df.columns})
    if "Principal" not in df: df["Principal"] = pd.to_numeric(df.iloc[:,0], errors="coerce")
    df["Principal"] = pd.to_numeric(df["Principal"], errors="coerce")
    df["InterestRate"] = pd.to_numeric(df.get("Interestrate", np.nan), errors="coerce")
    df["Currency"] = currency
    return df

# ------------------ Process stocks ------------------
def process_stocks(df):
    if df is None or df.empty: return pd.DataFrame()
    tickers = df["ticker"].dropna().unique().tolist()
    last = batch_last_close(tickers)
    rows = []
    for _, r in df.iterrows():
        ticker = r["ticker"]
        shares = pd.to_numeric(r.get("shares"), errors="coerce")
        avg_cost = pd.to_numeric(r.get("avg_cost"), errors="coerce")
        last_price = last.get(ticker, np.nan)

        cost_native = shares * avg_cost if pd.notna(shares) and pd.notna(avg_cost) else np.nan
        mv_native = shares * last_price if pd.notna(shares) and pd.notna(last_price) else np.nan
        currency = r.get("currency", "USD")

        if str(currency).upper() == "INR" or str(ticker).endswith(".NS"):
            mv_inr, mv_usd = mv_native, (mv_native / fx_rate if pd.notna(mv_native) else np.nan)
            cost_inr, cost_usd = cost_native, (cost_native / fx_rate if pd.notna(cost_native) else np.nan)
        else:
            mv_usd, mv_inr = mv_native, (mv_native * fx_rate if pd.notna(mv_native) else np.nan)
            cost_usd, cost_inr = cost_native, (cost_native * fx_rate if pd.notna(cost_native) else np.nan)

        rows.append({
            "Ticker": ticker,
            "Shares": shares,
            "Avg Cost (native)": avg_cost,
            "Cost INR": cost_inr, "Cost USD": cost_usd,
            "Market Value INR": mv_inr, "Market Value USD": mv_usd
        })
    return pd.DataFrame(rows)

# ------------------ Load all portfolios ------------------
india_hold = process_stocks(load_stock_csv(india_file, True))
us_hold = process_stocks(load_stock_csv(us_file, False))
bonds_inr_df = load_asset_csv(bonds_inr_file, "INR")
bonds_usd_df = load_asset_csv(bonds_usd_file, "USD")
fds_inr_df = load_asset_csv(fds_inr_file, "INR")
fds_usd_df = load_asset_csv(fds_usd_file, "USD")

# ------------------ Allocation summary ------------------
def segment_sums():
    india_inr = india_hold["Market Value INR"].sum() if not india_hold.empty else 0.0
    india_usd = india_inr / fx_rate
    us_usd = us_hold["Market Value USD"].sum() if not us_hold.empty else 0.0
    us_inr = us_usd * fx_rate

    bonds_inr = bonds_inr_df["Principal"].sum() if not bonds_inr_df.empty else 0.0
    bonds_usd = bonds_usd_df["Principal"].sum() if not bonds_usd_df.empty else 0.0
    fds_inr = fds_inr_df["Principal"].sum() if not fds_inr_df.empty else 0.0
    fds_usd = fds_usd_df["Principal"].sum() if not fds_usd_df.empty else 0.0

    return {
        "INR Stocks": (india_inr, india_usd),
        "USD Stocks": (us_inr, us_usd),
        "INR Bonds": (bonds_inr, bonds_inr/fx_rate),
        "USD Bonds": (bonds_usd*fx_rate, bonds_usd),
        "INR FDs": (fds_inr, fds_inr/fx_rate),
        "USD FDs": (fds_usd*fx_rate, fds_usd)
    }

segments = segment_sums()

# Allocation table
alloc_df = pd.DataFrame([
    {"Segment": k, "Amount INR": v[0], "Amount USD": v[1]} for k,v in segments.items()
])
alloc_df["Percent (%)"] = (alloc_df["Amount INR"] / alloc_df["Amount INR"].sum() * 100).round(2)
st.subheader("Allocation Split")
st.dataframe(alloc_df, use_container_width=True)

# Pie
pie = alt.Chart(alloc_df).mark_arc(innerRadius=50).encode(
    theta="Percent (%):Q", color="Segment:N", tooltip=["Segment","Percent (%)","Amount INR"]
)
st.altair_chart(pie, use_container_width=True)

# ------------------ ROI Planner ------------------
st.subheader("ROI Planner")

dep = inr_depr/100.0
r_india = india_growth/100.0 - dep
r_us = us_growth/100.0
r_b_inr = (bonds_inr_df["InterestRate"].mean()/100.0 if not bonds_inr_df.empty else 0.06) - dep
r_b_usd = (bonds_usd_df["InterestRate"].mean()/100.0 if not bonds_usd_df.empty else 0.03)
r_fd_inr = (fds_inr_df["InterestRate"].mean()/100.0 if not fds_inr_df.empty else 0.05) - dep
r_fd_usd = (fds_usd_df["InterestRate"].mean()/100.0 if not fds_usd_df.empty else 0.025)

# Current expected ROI
tot_usd = alloc_df["Amount USD"].sum()
weights = alloc_df.set_index("Segment")["Amount USD"]/tot_usd if tot_usd>0 else None
returns = {
    "INR Stocks": r_india, "USD Stocks": r_us,
    "INR Bonds": r_b_inr, "USD Bonds": r_b_usd,
    "INR FDs": r_fd_inr, "USD FDs": r_fd_usd
}
exp_roi = sum(weights[s]*returns[s] for s in returns) if weights is not None else 0
st.write(f"Expected ROI (USD terms): **{exp_roi*100:.2f}%** | Target: **{target_roi:.2f}%**")
