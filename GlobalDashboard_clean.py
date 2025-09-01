
# -*- coding: utf-8 -*-
"""
GlobalDashboard_clean.py
Single-file Streamlit app for Global Portfolio Dashboard — ready for deployment.
Features:
- Two portfolio uploads (India / US)
- Live P&L (Cost, Market Value, Unrealized P/L)
- Currency toggle (INR / USD)
- Safe Yahoo Finance downloader (won't hang Streamlit)
- Benchmarks and India/US cost-basis performance chart (user-selectable lookback)
- Estimation modules: Projection Settings, Bonds, Fixed Deposits, Target ROI Planner
- CSV / Excel exports
- Auto-refresh optional
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import altair as alt
from io import BytesIO
from datetime import date

st.set_page_config(page_title="Global Portfolio Dashboard", layout="wide")
st.title("📈 Global Portfolio Dashboard — Clean Deployment")

# ---------------------- AUTO REFRESH ----------------------
try:
    from streamlit_autorefresh import st_autorefresh
    _ = st_autorefresh(interval=5*60*1000, key="auto_refresh_5m")
except Exception:
    st.sidebar.caption("Tip: `pip install streamlit-autorefresh` to enable auto-refresh (5 min).")

# ---------------------- SIDEBAR: UPLOADS & SETTINGS ----------------------
st.sidebar.header("Uploads & Settings")
india_file = st.sidebar.file_uploader("Upload Indian Portfolio CSV (no .NS)", type="csv")
us_file = st.sidebar.file_uploader("Upload US Portfolio CSV", type="csv")
currency_choice = st.sidebar.radio("Display Currency", ["INR", "USD"], index=0)

# Lookback and chart toggle
lookback = st.sidebar.selectbox("Performance lookback period", ["3mo", "6mo", "1y", "2y"], index=2)
show_chart = st.sidebar.checkbox("Show Benchmarks & Portfolio Chart", value=True)

# ---------------------- SAFE DOWNLOAD UTILITIES ----------------------
@st.cache_data(ttl=300)
def safe_download(tickers, period="1y", interval="1d", group_by="ticker", threads=True):
    """
    Robust wrapper around yfinance.download that returns a DataFrame or empty DataFrame.
    Caches results for `ttl` seconds to prevent repeated network calls.
    """
    try:
        df = yf.download(tickers, period=period, interval=interval, progress=False,
                         group_by=group_by, threads=threads)
        if df is None:
            return pd.DataFrame()
        # If result is an empty DataFrame, return empty DataFrame
        if isinstance(df, pd.DataFrame) and df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def batch_last_close(symbols):
    """Fetch last close for many symbols quickly with safe_download. Returns dict(symbol->price)."""
    if not symbols:
        return {}
    out = {s: np.nan for s in symbols}
    # Try download for all symbols
    try:
        data = safe_download(symbols, period="5d", interval="1d", group_by="ticker", threads=True)
    except Exception:
        data = pd.DataFrame()
    # Helper extractor
    def extract_close(d, sym):
        try:
            if isinstance(d, pd.DataFrame) and isinstance(d.columns, pd.MultiIndex):
                # Multi-ticker: d[sym]["Close"]
                ser = d[sym]["Close"].dropna()
                if not ser.empty:
                    return float(ser.iloc[-1])
            if isinstance(d, pd.DataFrame) and "Close" in d.columns:
                ser = d["Close"].dropna()
                if not ser.empty:
                    return float(ser.iloc[-1])
        except Exception:
            pass
        # Fallback to ticker.history
        try:
            h = yf.Ticker(sym).history(period="5d", interval="1d")
            if not h.empty:
                return float(h["Close"].dropna().iloc[-1])
        except Exception:
            pass
        return np.nan

    if isinstance(data, pd.DataFrame) and not data.empty:
        for s in symbols:
            out[s] = extract_close(data, s)
    else:
        for s in symbols:
            out[s] = extract_close(pd.DataFrame(), s)
    return out

# ---------------------- PORTFOLIO LOADING & PROCESSING ----------------------
def load_csv_portfolio(uploaded_file, is_india=True):
    if uploaded_file is None:
        return None
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Ticker":"ticker","Quantity":"shares","AvgCost":"avg_cost"})
    # ensure types
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["avg_cost"] = pd.to_numeric(df["avg_cost"], errors="coerce")
    if is_india:
        df["ticker"] = df["ticker"].astype(str).apply(lambda x: x if x.endswith(".NS") else f"{x}.NS")
        df["currency"] = "INR"
    else:
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["currency"] = "USD"
    return df

def process_portfolio(df):
    """Compute cost, last price, market value, pnl for portfolio DataFrame"""
    if df is None or df.empty:
        return pd.DataFrame()
    syms = df["ticker"].dropna().unique().tolist()
    last_map = batch_last_close(syms)
    rows = []
    for _, r in df.iterrows():
        sym = r["ticker"]
        shares = pd.to_numeric(r.get("shares"), errors="coerce")
        avg_cost = pd.to_numeric(r.get("avg_cost"), errors="coerce")
        last_price = last_map.get(sym, np.nan)
        cost = shares * avg_cost if pd.notna(shares) and pd.notna(avg_cost) else np.nan
        mv = shares * last_price if pd.notna(shares) and pd.notna(last_price) else np.nan
        pl = mv - cost if pd.notna(mv) and pd.notna(cost) else np.nan
        pl_pct = (pl / cost * 100) if pd.notna(pl) and pd.notna(cost) and cost>0 else np.nan
        rows.append({
            "Ticker": sym,
            "Shares": shares,
            "Avg Cost": avg_cost,
            "Cost": cost,
            "Last Price": last_price,
            "Market Value": mv,
            "Unrealized P/L": pl,
            "Unrealized P/L %": pl_pct
        })
    return pd.DataFrame(rows)

# Currency conversion based on display currency choice
@st.cache_data(ttl=600)
def get_fx_rate():
    try:
        fx = yf.Ticker("USDINR=X").fast_info.get("lastPrice", None)
        if fx is None or (isinstance(fx, float) and np.isnan(fx)):
            d = safe_download("USDINR=X", period="1d", interval="5m")
            if not d.empty:
                if isinstance(d, pd.DataFrame) and "Close" in d.columns:
                    fx = float(d["Close"].dropna().iloc[-1])
        return float(fx) if fx is not None else 83.0
    except Exception:
        return 83.0

fx_rate = get_fx_rate()

def convert_currency(df, to="INR"):
    if df is None or df.empty:
        return df
    out = df.copy()
    if to == "USD":
        out["Cost"] = out.apply(lambda r: r["Cost"]/fx_rate if str(r["Ticker"]).endswith(".NS") else r["Cost"], axis=1)
        out["Market Value"] = out.apply(lambda r: r["Market Value"]/fx_rate if str(r["Ticker"]).endswith(".NS") else r["Market Value"], axis=1)
    else:
        out["Cost"] = out.apply(lambda r: r["Cost"] if str(r["Ticker"]).endswith(".NS") else r["Cost"]*fx_rate, axis=1)
        out["Market Value"] = out.apply(lambda r: r["Market Value"] if str(r["Ticker"]).endswith(".NS") else r["Market Value"]*fx_rate, axis=1)
    out["Cost"] = pd.to_numeric(out["Cost"], errors="coerce")
    out["Market Value"] = pd.to_numeric(out["Market Value"], errors="coerce")
    out["Unrealized P/L"] = out["Market Value"] - out["Cost"]
    out["Unrealized P/L %"] = ((out["Unrealized P/L"] / out["Cost"]) * 100).round(2)
    return out

# ---------------------- BUILD AND DISPLAY PORTFOLIOS ----------------------
india_raw = load_csv_portfolio(india_file, is_india=True)
us_raw = load_csv_portfolio(us_file, is_india=False)

india_proc = process_portfolio(india_raw) if india_raw is not None else pd.DataFrame()
us_proc = process_portfolio(us_raw) if us_raw is not None else pd.DataFrame()

india_proc = convert_currency(india_proc, to=currency_choice) if not india_proc.empty else india_proc
us_proc = convert_currency(us_proc, to=currency_choice) if not us_proc.empty else us_proc

# combined
if not india_proc.empty and not us_proc.empty:
    combined_proc = pd.concat([india_proc, us_proc], ignore_index=True)
elif not india_proc.empty:
    combined_proc = india_proc.copy()
elif not us_proc.empty:
    combined_proc = us_proc.copy()
else:
    combined_proc = pd.DataFrame()

# ---------------------- SUMMARY METRICS ----------------------
def show_summary(df, title):
    if df is None or df.empty:
        return
    total_cost = df["Cost"].sum()
    total_mv = df["Market Value"].sum()
    total_pl = total_mv - total_cost
    total_pl_pct = (total_pl / total_cost * 100) if total_cost>0 else np.nan
    c1,c2,c3 = st.columns(3)
    c1.metric(f"{title} - Total Cost ({currency_choice})", f"{total_cost:,.2f}")
    c2.metric(f"{title} - Market Value ({currency_choice})", f"{total_mv:,.2f}")
    c3.metric(f"{title} - Unrealized P/L ({currency_choice})", f"{total_pl:,.2f}", delta=f"{total_pl_pct:.2f}%")

st.subheader("Portfolio Summary")
show_summary(combined_proc, "Overall")
show_summary(india_proc, "India")
show_summary(us_proc, "US")

# ---------------------- BENCHMARKS & PORTFOLIO PERFORMANCE (cost-basis) ----------------------
def bench_series(sym, period=lookback):
    df = safe_download(sym, period=period, interval="1d")
    if df.empty:
        return pd.Series(dtype=float)
    # handle when single ticker returns DataFrame with 'Close' column
    if "Close" in df.columns:
        ser = df["Close"].pct_change().add(1).cumprod() - 1
        ser.name = sym
        return ser
    # multiindex case unlikely for single ticker, but try to extract
    try:
        ser = df[sym]["Close"].pct_change().add(1).cumprod() - 1
        ser.name = sym
        return ser
    except Exception:
        return pd.Series(dtype=float)

def build_cost_basis_series(df_proc, label, period=lookback):
    if df_proc is None or df_proc.empty:
        return None
    total_cost = df_proc["Cost"].sum()
    if not pd.notna(total_cost) or total_cost<=0:
        return None
    parts = []
    for _, row in df_proc.iterrows():
        t = row["Ticker"]
        avg_cost_native = row.get("Avg Cost", None)
        if not pd.notna(avg_cost_native) or avg_cost_native<=0:
            continue
        weight = (row["Cost"] / total_cost) if pd.notna(row["Cost"]) else 0.0
        if weight<=0:
            continue
        hist = safe_download(t, period=period, interval="1d")
        if hist.empty:
            continue
        # extract close series robustly
        if "Close" in hist.columns:
            close = hist["Close"]
        else:
            try:
                close = hist[t]["Close"]
            except Exception:
                continue
        contrib = (close / float(avg_cost_native) - 1.0) * float(weight)
        contrib.name = t
        parts.append(contrib)
    if not parts:
        return None
    df_aligned = pd.concat(parts, axis=1).fillna(0.0)
    series = df_aligned.sum(axis=1)
    series.name = label
    return series

# Plot chart if enabled
if show_chart:
    series_list = []
    # Benchmarks
    for name, sym in {"S&P 500":"^GSPC","NASDAQ":"^IXIC","NIFTY50":"^NSEI"}.items():
        s = bench_series(sym, period=lookback)
        if not s.empty:
            s.name = name
            series_list.append(s)
    # India & US cost-basis series
    india_series = build_cost_basis_series(india_proc, "India Portfolio", period=lookback)
    us_series = build_cost_basis_series(us_proc, "US Portfolio", period=lookback)
    if india_series is not None:
        series_list.append(india_series)
    if us_series is not None:
        series_list.append(us_series)

    if series_list:
        chart_df = pd.concat(series_list, axis=1).reset_index().rename(columns={"index":"Date"})
        long_df = chart_df.melt(id_vars="Date", var_name="Series", value_name="Return")
        ch = (alt.Chart(long_df).mark_line().encode(
                x="Date:T", y=alt.Y("Return:Q", axis=alt.Axis(format="%")),
                color="Series:N", tooltip=["Date:T","Series:N", alt.Tooltip("Return:Q", format=".2%")]
            ).properties(title="Benchmark & Portfolio Performance (cost-basis)"))
        st.altair_chart(ch, use_container_width=True)
    else:
        st.info("No benchmark/portfolio series available for plotting.")

# Consistency check
def check_alignment(proc_df, series, label):
    if proc_df is None or proc_df.empty or series is None or series.empty: 
        return
    tot_cost = proc_df["Cost"].sum()
    tot_mv = proc_df["Market Value"].sum()
    if not (pd.notna(tot_cost) and tot_cost>0 and pd.notna(tot_mv)):
        return
    summary_ret = (tot_mv - tot_cost) / tot_cost
    chart_ret = series.dropna().iloc[-1]
    st.caption(f"🔎 {label}: Summary = {summary_ret:.2%} | Chart = {chart_ret:.2%} | Δ = {(summary_ret - chart_ret):.2%}")

check_alignment(india_proc, build_cost_basis_series(india_proc, "India Portfolio", period=lookback), "India Portfolio")
check_alignment(us_proc, build_cost_basis_series(us_proc, "US Portfolio", period=lookback), "US Portfolio")

# ---------------------- ESTIMATION MODULES (SIDE MULTISELECT) ----------------------
st.sidebar.header("Estimation Modules")
modules = st.sidebar.multiselect("Enable modules", ["Projection Settings","Bonds","Fixed Deposits","Target ROI Planner"], default=["Projection Settings","Target ROI Planner"])

# Projection Settings: sliders for growth
india_growth = 12.0
us_growth = 8.0
target_roi = 10.0
if "Projection Settings" in modules:
    st.sidebar.subheader("Projection Settings")
    india_growth = st.sidebar.slider("India stocks future growth rate (%)", -50.0, 50.0, 12.0, 0.1)
    us_growth = st.sidebar.slider("US stocks future growth rate (%)", -50.0, 50.0, 8.0, 0.1)
    target_roi = st.sidebar.slider("Overall expected ROI target (%)", -50.0, 50.0, 10.0, 0.1)

# Bonds module: collect rows in session state and allow CSV download
if "Bonds" in modules:
    st.subheader("Bonds — Enter Holdings")
    if "bonds" not in st.session_state:
        st.session_state.bonds = []
    with st.form("bonds_form", clear_on_submit=True):
        b_principal = st.number_input("Principal Amount (INR)", min_value=0.0, value=100000.0, step=1000.0)
        b_date = st.date_input("Date of Purchase", value=date.today())
        b_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=6.5, step=0.01)
        b_add = st.form_submit_button("Add Bond")
    if b_add:
        st.session_state.bonds.append({"Principal (INR)": float(b_principal), "Purchase Date": str(b_date), "Interest Rate (%)": float(b_rate)})
    bonds_df = pd.DataFrame(st.session_state.bonds)
    st.dataframe(bonds_df, use_container_width=True)
    st.download_button("Download Bonds CSV", data=bonds_df.to_csv(index=False), file_name="bonds.csv", mime="text/csv")

# Fixed Deposits module
if "Fixed Deposits" in modules:
    st.subheader("Fixed Deposits — Enter Holdings")
    if "fds" not in st.session_state:
        st.session_state.fds = []
    with st.form("fds_form", clear_on_submit=True):
        f_principal = st.number_input("Principal Amount (INR)", min_value=0.0, value=50000.0, step=1000.0)
        f_date = st.date_input("Date of Deposit", value=date.today())
        f_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=5.5, step=0.01)
        f_add = st.form_submit_button("Add FD")
    if f_add:
        st.session_state.fds.append({"Principal (INR)": float(f_principal), "Deposit Date": str(f_date), "Interest Rate (%)": float(f_rate)})
    fds_df = pd.DataFrame(st.session_state.fds)
    st.dataframe(fds_df, use_container_width=True)
    st.download_button("Download Fixed Deposits CSV", data=fds_df.to_csv(index=False), file_name="fixed_deposits.csv", mime="text/csv")

# ---------------------- TARGET ROI PLANNER ----------------------
if "Target ROI Planner" in modules:
    st.subheader("🎯 Target ROI Planner")
    total_mv_india = 0.0 if india_proc is None or india_proc.empty else float(india_proc["Market Value"].sum())
    total_mv_us = 0.0 if us_proc is None or us_proc.empty else float(us_proc["Market Value"].sum())
    total_mv = total_mv_india + total_mv_us
    st.write(f"Current Market Value — India: **{total_mv_india:,.2f} {currency_choice}**, US: **{total_mv_us:,.2f} {currency_choice}**, Total: **{total_mv:,.2f} {currency_choice}**")

    # Expected ROI from current allocation & user growth sliders
    weight_india = (total_mv_india / total_mv) if total_mv>0 else 0.0
    weight_us = 1.0 - weight_india
    expected_roi_current = weight_india*(india_growth/100.0) + weight_us*(us_growth/100.0)
    st.metric("Expected ROI from current allocation (annual)", f"{expected_roi_current*100:.2f}%")

    # Given target ROI (slider), compute suggested split (fraction to India)
    gI = india_growth/100.0
    gU = us_growth/100.0
    tR = target_roi/100.0
    split_india = np.nan
    if abs(gI - gU) < 1e-9:
        split_india = 0.5
    else:
        split_india = (tR - gU) / (gI - gU)
    split_india = float(np.clip(split_india, 0.0, 1.0))
    split_us = 1.0 - split_india
    amt_india = split_india * total_mv
    amt_us = split_us * total_mv

    st.write(f"Suggested split to target **{target_roi:.2f}%** ROI: India **{split_india*100:.2f}%**, US **{split_us*100:.2f}%**")
    pie_df = pd.DataFrame({"Segment":["India","US"], "Allocation %":[split_india*100, split_us*100], "Amount":[amt_india, amt_us]})
    pie = (alt.Chart(pie_df).mark_arc().encode(theta="Allocation %:Q", color="Segment:N", tooltip=["Segment","Allocation %:Q","Amount:Q"]).properties(title="Suggested Allocation Split"))
    st.altair_chart(pie, use_container_width=True)

# ---------------------- EXPORTS ----------------------
# per-tab CSVs and combined Excel
if not india_proc.empty:
    st.download_button("Download India CSV", data=india_proc.to_csv(index=False), file_name="india_portfolio_processed.csv", mime="text/csv")
if not us_proc.empty:
    st.download_button("Download US CSV", data=us_proc.to_csv(index=False), file_name="us_portfolio_processed.csv", mime="text/csv")

if not combined_proc.empty:
    # Excel
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        if not india_proc.empty: india_proc.to_excel(writer, sheet_name="India", index=False)
        if not us_proc.empty: us_proc.to_excel(writer, sheet_name="US", index=False)
        combined_proc.to_excel(writer, sheet_name="Combined", index=False)
    st.sidebar.download_button("Download Excel (All)", data=out.getvalue(), file_name="portfolio_all.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# End of script
