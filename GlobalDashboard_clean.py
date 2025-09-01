# -*- coding: utf-8 -*-
"""
GlobalDashboard_final.py
Complete Streamlit app with:
- India & US stock portfolios (CSV uploads)
- Bonds & Fixed Deposits (CSV uploads for INR & USD; manual entry supported)
- Safe Yahoo downloads with selectable lookback (including 1mo)
- INR depreciation slider (0-10%/yr) that adjusts USD-equivalent returns
- Allocation split table & pie chart with amounts in INR, USD and %
- Target ROI planner using user-provided growth sliders and bond/FD rates
- CSV/Excel exports, auto-refresh optional
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
india_file = st.sidebar.file_uploader("Upload Indian Portfolio CSV (Ticker,Quantity,AvgCost) — no .NS", type=["csv"], key="india_csv")
us_file = st.sidebar.file_uploader("Upload US Portfolio CSV (Ticker,Quantity,AvgCost)", type=["csv"], key="us_csv")

# Bonds & FDs CSV uploads (INR / USD separate)
bonds_inr_file = st.sidebar.file_uploader("Upload Bonds (INR) CSV (Principal,Date,InterestRate%)", type=["csv"], key="bonds_inr")
bonds_usd_file = st.sidebar.file_uploader("Upload Bonds (USD) CSV (Principal,Date,InterestRate%)", type=["csv"], key="bonds_usd")
fds_inr_file = st.sidebar.file_uploader("Upload Fixed Deposits (INR) CSV (Principal,Date,InterestRate%)", type=["csv"], key="fds_inr")
fds_usd_file = st.sidebar.file_uploader("Upload Fixed Deposits (USD) CSV (Principal,Date,InterestRate%)", type=["csv"], key="fds_usd")

# Display currency and lookback
display_currency = st.sidebar.radio("Display currency", ["INR", "USD"], index=0)
lookback = st.sidebar.selectbox("Performance lookback period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
show_chart = st.sidebar.checkbox("Show Benchmarks & Portfolio Chart", value=True)

# Growth sliders in sidebar
st.sidebar.subheader("Projection Inputs")
india_growth = st.sidebar.slider("India stocks future growth rate (%)", -50.0, 50.0, 12.0, 0.1)
us_growth = st.sidebar.slider("US stocks future growth rate (%)", -50.0, 50.0, 8.0, 0.1)
target_roi = st.sidebar.slider("Target overall ROI (%)", -50.0, 50.0, 10.0, 0.1)

# INR depreciation slider 0-10%/yr
inr_depr = st.sidebar.slider("INR depreciation vs USD (% per year)", 0.0, 10.0, 3.0, 0.1)

# Auto-refresh optional
try:
    from streamlit_autorefresh import st_autorefresh
    _ = st_autorefresh(interval=5 * 60 * 1000, key="autorefresh_final")
except Exception:
    pass

# ------------------ Utilities: safe downloads & extraction ------------------
@st.cache_data(ttl=300)
def safe_download(tickers, period="1y", interval="1d", group_by="ticker", threads=True):
    """Return DataFrame or empty DataFrame; doesn't block Streamlit on errors."""
    try:
        df = yf.download(tickers, period=period, interval=interval, progress=False, group_by=group_by, threads=threads)
        if df is None:
            return pd.DataFrame()
        if isinstance(df, pd.DataFrame) and df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def batch_last_close(symbols):
    """Return dict symbol -> last close price (robust for single or multiple tickers)."""
    if not symbols:
        return {}
    out = {s: np.nan for s in symbols}
    data = safe_download(symbols, period="5d", interval="1d", group_by="ticker", threads=True)
    def extract_close(d, sym):
        try:
            if isinstance(d, pd.DataFrame) and isinstance(d.columns, pd.MultiIndex):
                ser = d[sym]["Close"].dropna()
                if not ser.empty: return float(ser.iloc[-1])
            if isinstance(d, pd.DataFrame) and "Close" in d.columns:
                ser = d["Close"].dropna()
                if not ser.empty: return float(ser.iloc[-1])
        except Exception:
            pass
        # fallback
        try:
            h = yf.Ticker(sym).history(period="5d", interval="1d")
            if not h.empty: return float(h["Close"].dropna().iloc[-1])
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

# FX rate (USD/INR)
@st.cache_data(ttl=600)
def get_fx_rate():
    try:
        finf = yf.Ticker("USDINR=X").fast_info
        if finf and "lastPrice" in finf and finf["lastPrice"] is not None:
            return float(finf["lastPrice"])
    except Exception:
        pass
    # fallback to daily
    d = safe_download("USDINR=X", period="1d", interval="5m")
    try:
        if not d.empty and "Close" in d.columns:
            return float(d["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return 83.0

fx_rate = get_fx_rate()  # INR per 1 USD

# ------------------ Portfolio CSV loaders ------------------
def load_stock_csv(f, is_india=True):
    if f is None: return None
    df = pd.read_csv(f)
    df.columns = [c.strip() for c in df.columns]
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
    """Bonds/FD CSV loader; expected columns: Principal, Date, InterestRate (in %)"""
    if f is None: return pd.DataFrame()
    df = pd.read_csv(f)
    df.columns = [c.strip() for c in df.columns]
    # normalize names
    mapping = {}
    for col in df.columns:
        if col.lower().startswith("principal"): mapping[col] = "Principal"
        if "date" in col.lower(): mapping[col] = "Date"
        if "interest" in col.lower(): mapping[col] = "InterestRate"
    df = df.rename(columns=mapping)
    if "Principal" not in df.columns: df["Principal"] = pd.to_numeric(df.iloc[:,0], errors="coerce")
    df["Principal"] = pd.to_numeric(df["Principal"], errors="coerce")
    df["InterestRate"] = pd.to_numeric(df.get("InterestRate", np.nan), errors="coerce")
    df["Currency"] = currency
    return df

# Manual entry forms for bonds/fds if user prefers
def manual_add_asset(session_key, currency, label):
    if session_key not in st.session_state:
        st.session_state[session_key] = []
    with st.form(f"form_{session_key}", clear_on_submit=True):
        principal = st.number_input(f"{label} Principal ({currency})", min_value=0.0, value=100000.0)
        dt = st.date_input(f"{label} Date", value=date.today())
        rate = st.number_input(f"{label} Interest Rate (%)", min_value=0.0, value=6.5)
        submitted = st.form_submit_button("Add")
    if submitted:
        st.session_state[session_key].append({"Principal": float(principal), "Date": str(dt), "InterestRate": float(rate), "Currency": currency})

# ------------------ Process stocks into holdings with Market Values (native & converted) ------------------
def process_stocks(df):
    """Return dataframe with native market values and converted INR/USD amounts."""
    if df is None or df.empty:
        return pd.DataFrame()
    tickers = df["ticker"].dropna().unique().tolist()
    last = batch_last_close(tickers)
    rows = []
    for _, r in df.iterrows():
        ticker = r["ticker"]
        shares = float(r["shares"]) if pd.notna(r["shares']) else 0.0 if False else (r["shares"] if False else r["shares"])
        # (Above structure avoids linter complaining; ensure numeric)
        shares = pd.to_numeric(r["shares"], errors="coerce")
        avg_cost = pd.to_numeric(r["avg_cost"], errors="coerce")
        last_price = last.get(ticker, np.nan)
        cost_native = shares * avg_cost if pd.notna(shares) and pd.notna(avg_cost) else np.nan
        mv_native = shares * last_price if pd.notna(shares) and pd.notna(last_price) else np.nan
        currency = r.get("currency","USD")
        # convert to both INR/USD
        if str(currency).upper() == "INR" or ticker.endswith(".NS"):
            mv_inr = mv_native
            mv_usd = (mv_native / fx_rate) if pd.notna(mv_native) else np.nan
            cost_inr = cost_native
            cost_usd = (cost_native / fx_rate) if pd.notna(cost_native) else np.nan
        else:
            mv_usd = mv_native
            mv_inr = (mv_native * fx_rate) if pd.notna(mv_native) else np.nan
            cost_usd = cost_native
            cost_inr = (cost_native * fx_rate) if pd.notna(cost_native) else np.nan

        rows.append({
            "Ticker": ticker,
            "Shares": shares,
            "Avg Cost (native)": avg_cost,
            "Cost (native)": cost_native,
            "Last Price (native)": last_price,
            "Market Value (native)": mv_native,
            "Cost INR": cost_inr,
            "Cost USD": cost_usd,
            "Market Value INR": mv_inr,
            "Market Value USD": mv_usd
        })
    return pd.DataFrame(rows)

# ------------------ Load and process uploaded CSVs ------------------
india_raw = load_stock_csv(india_file, is_india=True) if india_file else pd.DataFrame()
us_raw = load_stock_csv(us_file, is_india=False) if us_file else pd.DataFrame()

india_hold = process_stocks(india_raw) if not india_raw.empty else pd.DataFrame()
us_hold = process_stocks(us_raw) if not us_raw.empty else pd.DataFrame()

# Bonds & FDs from CSVs
bonds_inr_df = load_asset_csv(bonds_inr_file, currency="INR")
bonds_usd_df = load_asset_csv(bonds_usd_file, currency="USD")
fds_inr_df = load_asset_csv(fds_inr_file, currency="INR")
fds_usd_df = load_asset_csv(fds_usd_file, currency="USD")

# Manual add forms appear in main area if user wants
st.sidebar.subheader("Manual Bond/FD entry (optional)")
manual_add_asset("bonds_inr_manual", "INR", "Bond (INR)")
manual_add_asset("bonds_usd_manual", "USD", "Bond (USD)")
manual_add_asset("fds_inr_manual", "INR", "FD (INR)")
manual_add_asset("fds_usd_manual", "USD", "FD (USD)")

# Combine manual entries into DataFrames
bonds_inr_manual = pd.DataFrame(st.session_state.get("bonds_inr_manual", []))
bonds_usd_manual = pd.DataFrame(st.session_state.get("bonds_usd_manual", []))
fds_inr_manual = pd.DataFrame(st.session_state.get("fds_inr_manual", []))
fds_usd_manual = pd.DataFrame(st.session_state.get("fds_usd_manual", []))

# Merge uploaded CSVs and manual entries
if not bonds_inr_manual.empty:
    bonds_inr_df = pd.concat([bonds_inr_df, bonds_inr_manual], ignore_index=True) if not bonds_inr_df.empty else bonds_inr_manual.copy()
if not bonds_usd_manual.empty:
    bonds_usd_df = pd.concat([bonds_usd_df, bonds_usd_manual], ignore_index=True) if not bonds_usd_df.empty else bonds_usd_manual.copy()
if not fds_inr_manual.empty:
    fds_inr_df = pd.concat([fds_inr_df, fds_inr_manual], ignore_index=True) if not fds_inr_df.empty else fds_inr_manual.copy()
if not fds_usd_manual.empty:
    fds_usd_df = pd.concat([fds_usd_df, fds_usd_manual], ignore_index=True) if not fds_usd_df.empty else fds_usd_manual.copy()

# ------------------ Summaries: amounts per segment ------------------
def segment_sums():
    # Stocks
    india_stock_inr = 0.0 if india_hold.empty else float(india_hold["Market Value INR"].sum(skipna=True))
    india_stock_usd = (india_stock_inr / fx_rate) if india_stock_inr and fx_rate else 0.0
    us_stock_usd = 0.0 if us_hold.empty else float(us_hold["Market Value USD"].sum(skipna=True))
    us_stock_inr = us_stock_usd * fx_rate if us_stock_usd and fx_rate else 0.0

    # Bonds and FDs totals (by currency)
    bonds_inr_total = 0.0 if bonds_inr_df.empty else float(bonds_inr_df["Principal"].sum(skipna=True))
    bonds_usd_total = 0.0 if bonds_usd_df.empty else float(bonds_usd_df["Principal"].sum(skipna=True))
    fds_inr_total = 0.0 if fds_inr_df.empty else float(fds_inr_df["Principal"].sum(skipna=True))
    fds_usd_total = 0.0 if fds_usd_df.empty else float(fds_usd_df["Principal"].sum(skipna=True))

    # Convert bonds/fds between currencies
    bonds_inr_inr = bonds_inr_total
    bonds_inr_usd = bonds_inr_total / fx_rate if fx_rate else 0.0
    bonds_usd_usd = bonds_usd_total
    bonds_usd_inr = bonds_usd_total * fx_rate if fx_rate else 0.0

    fds_inr_inr = fds_inr_total
    fds_inr_usd = fds_inr_total / fx_rate if fx_rate else 0.0
    fds_usd_usd = fds_usd_total
    fds_usd_inr = fds_usd_total * fx_rate if fx_rate else 0.0

    # Total sums in INR and USD
    total_inr = (india_stock_inr + us_stock_inr + bonds_inr_inr + bonds_usd_inr + fds_inr_inr + fds_usd_inr)
    total_usd = (india_stock_usd + us_stock_usd + bonds_inr_usd + bonds_usd_usd + fds_inr_usd + fds_usd_usd)

    return {
        "india_stock_inr": india_stock_inr, "india_stock_usd": india_stock_usd,
        "us_stock_inr": us_stock_inr, "us_stock_usd": us_stock_usd,
        "bonds_inr_inr": bonds_inr_inr, "bonds_inr_usd": bonds_inr_usd,
        "bonds_usd_inr": bonds_usd_inr, "bonds_usd_usd": bonds_usd_usd,
        "fds_inr_inr": fds_inr_inr, "fds_inr_usd": fds_inr_usd,
        "fds_usd_inr": fds_usd_inr, "fds_usd_usd": fds_usd_usd,
        "total_inr": total_inr, "total_usd": total_usd
    }

segments = segment_sums()

# ------------------ Display allocation split (table + pie) ------------------
st.subheader("Current Allocation Split — amounts & percentages (by segment)")

alloc_rows = [
    {"Segment": "USD Stocks", "Amount USD": segments["us_stock_usd"], "Amount INR": segments["us_stock_inr"]},
    {"Segment": "INR Stocks", "Amount USD": segments["india_stock_usd"], "Amount INR": segments["india_stock_inr"]},
    {"Segment": "USD Bonds", "Amount USD": segments["bonds_usd_usd"], "Amount INR": segments["bonds_usd_inr"]},
    {"Segment": "INR Bonds", "Amount USD": segments["bonds_inr_usd"], "Amount INR": segments["bonds_inr_inr"]},
    {"Segment": "USD FDs", "Amount USD": segments["fds_usd_usd"], "Amount INR": segments["fds_usd_inr"]},
    {"Segment": "INR FDs", "Amount USD": segments["fds_inr_usd"], "Amount INR": segments["fds_inr_inr"]},
]
alloc_df = pd.DataFrame(alloc_rows)
# compute totals and percent
total_inr = segments["total_inr"] if segments["total_inr"] else 0.0
total_usd = segments["total_usd"] if segments["total_usd"] else 0.0
# percent by INR basis (use INR totals to get % share)
alloc_df["Percent (%)"] = alloc_df["Amount INR"].apply(lambda x: (x/total_inr*100) if total_inr>0 else 0.0)
alloc_df["Amount INR"] = alloc_df["Amount INR"].round(2)
alloc_df["Amount USD"] = alloc_df["Amount USD"].round(4)
alloc_df["Percent (%)"] = alloc_df["Percent (%)"].round(2)

st.dataframe(alloc_df, use_container_width=True)

# Pie chart using INR amounts (so chart sums to 100%)
pie_df = alloc_df[["Segment", "Percent (%)", "Amount INR"]].copy()
pie = alt.Chart(pie_df).mark_arc(innerRadius=50).encode(
    theta=alt.Theta(field="Percent (%)", type="quantitative"),
    color=alt.Color(field="Segment", type="nominal"),
    tooltip=["Segment", alt.Tooltip("Percent (%)", format=".2f"), alt.Tooltip("Amount INR", format=".2f")]
).properties(height=350, title="Allocation Split (by INR amount)")
st.altair_chart(pie, use_container_width=True)

# ------------------ Summary metrics (stocks) ------------------
def display_stock_summaries():
    st.subheader("Holdings Summary")
    c1,c2 = st.columns(2)
    with c1:
        st.write("India holdings (native INR / converted USD):")
        if india_hold.empty:
            st.write("No India stock data.")
        else:
            display_df = india_hold[["Ticker","Shares","Avg Cost (native)","Market Value INR","Market Value USD"]].copy()
            display_df.columns = ["Ticker","Shares","Avg Cost (INR)","Market Value (INR)","Market Value (USD)"]
            st.dataframe(display_df, use_container_width=True)
    with c2:
        st.write("US holdings (native USD / converted INR):")
        if us_hold.empty:
            st.write("No US stock data.")
        else:
            display_df = us_hold[["Ticker","Shares","Avg Cost (native)","Market Value USD","Market Value INR"]].copy()
            display_df.columns = ["Ticker","Shares","Avg Cost (USD)","Market Value (USD)","Market Value (INR)"]
            st.dataframe(display_df, use_container_width=True)

display_stock_summaries()

# ------------------ Bonds & FDs display & download ------------------
st.subheader("Bonds & Fixed Deposits (INR & USD)")
col1, col2 = st.columns(2)
with col1:
    st.write("Bonds (INR):")
    if bonds_inr_df.empty:
        st.write("No INR bonds uploaded/entered.")
    else:
        st.dataframe(bonds_inr_df, use_container_width=True)
    st.download_button("Download Bonds INR CSV", data=bonds_inr_df.to_csv(index=False), file_name="bonds_inr.csv", mime="text/csv")
with col2:
    st.write("Bonds (USD):")
    if bonds_usd_df.empty:
        st.write("No USD bonds uploaded/entered.")
    else:
        st.dataframe(bonds_usd_df, use_container_width=True)
    st.download_button("Download Bonds USD CSV", data=bonds_usd_df.to_csv(index=False), file_name="bonds_usd.csv", mime="text/csv")

col3, col4 = st.columns(2)
with col3:
    st.write("FDs (INR):")
    if fds_inr_df.empty:
        st.write("No INR FDs uploaded/entered.")
    else:
        st.dataframe(fds_inr_df, use_container_width=True)
    st.download_button("Download FDs INR CSV", data=fds_inr_df.to_csv(index=False), file_name="fds_inr.csv", mime="text/csv")
with col4:
    st.write("FDs (USD):")
    if fds_usd_df.empty:
        st.write("No USD FDs uploaded/entered.")
    else:
        st.dataframe(fds_usd_df, use_container_width=True)
    st.download_button("Download FDs USD CSV", data=fds_usd_df.to_csv(index=False), file_name="fds_usd.csv", mime="text/csv")

# ------------------ Benchmark + cost-basis chart ------------------
if show_chart:
    st.subheader("Benchmark & Portfolio Performance (cost-basis)")
    # Benchmarks function: support lookback including 1mo
    def bench_series(sym, period=lookback):
        df = safe_download(sym, period=period, interval="1d")
        if df.empty: return pd.Series(dtype=float)
        # single ticker may return DataFrame with Close column
        try:
            if "Close" in df.columns:
                s = df["Close"].pct_change().add(1).cumprod() - 1
                s.name = sym
                return s
            else:
                s = df[sym]["Close"].pct_change().add(1).cumprod() - 1
                s.name = sym
                return s
        except Exception:
            return pd.Series(dtype=float)

    # Build list of series: benchmarks + India/US cost-basis
    series_list = []
    for name, sym in {"S&P 500":"^GSPC","NASDAQ":"^IXIC","NIFTY50":"^NSEI"}.items():
        s = bench_series(sym, period=lookback)
        if not s.empty:
            s.name = name
            series_list.append(s)

    def build_cost_basis_series(df_proc, label, period=lookback):
        if df_proc is None or df_proc.empty:
            return None
        total_cost_inr = df_proc["Cost INR"].sum() if "Cost INR" in df_proc.columns else df_proc["Cost INR"].sum() if "Cost INR" in df_proc else df_proc["Cost INR"].sum()
        # We'll use avg cost native and weight by cost in display currency (INR)
        total_cost = df_proc["Cost INR"].sum()
        if not pd.notna(total_cost) or total_cost <= 0:
            return None
        parts = []
        for _, r in df_proc.iterrows():
            ticker = r["Ticker"]
            # avg cost native:
            avg_native = r["Avg Cost (native)"] if "Avg Cost (native)" in r else r.get("Avg Cost (native)", np.nan)
            weight = (r["Cost INR"] / total_cost) if pd.notna(r["Cost INR"]) else 0.0
            if not (pd.notna(avg_native) and avg_native>0 and weight>0):
                continue
            hist = safe_download(ticker, period=period, interval="1d")
            if hist.empty: continue
            close = hist["Close"] if "Close" in hist.columns else hist[ticker]["Close"]
            contrib = (close / float(avg_native) - 1.0) * float(weight)
            contrib.name = ticker
            parts.append(contrib)
        if not parts: return None
        df_aligned = pd.concat(parts, axis=1).fillna(0.0)
        series = df_aligned.sum(axis=1)
        series.name = label
        return series

    # Prepare dataframes with required fields for builder
    # For builder we need columns: Ticker, Avg Cost (native), Cost INR
    # Prepare india_proc_small and us_proc_small
    def prep_for_builder(df_hold, is_india=True):
        if df_hold is None or df_hold.empty:
            return pd.DataFrame()
        out = pd.DataFrame()
        out["Ticker"] = df_hold["Ticker"]
        out["Avg Cost (native)"] = df_hold["Avg Cost (native)"]
        out["Cost INR"] = df_hold["Market Value INR"].where(pd.notna(df_hold["Market Value INR"]), df_hold["Cost INR"])
        return out

    # But our earlier process_stocks gave different column names. Let's build df_proc with required fields:
    def build_df_proc_from_hold(hold_df, is_india=True):
        if hold_df is None or hold_df.empty:
            return pd.DataFrame()
        df = pd.DataFrame()
        df["Ticker"] = hold_df["Ticker"]
        df["Avg Cost (native)"] = hold_df["Avg Cost (native)"]
        df["Cost INR"] = hold_df["Cost INR"]
        return df

    india_builder = build_df_proc_from_hold(india_hold, is_india=True)
    us_builder = build_df_proc_from_hold(us_hold, is_india=False)

    ind_series = build_cost_basis_series(india_builder, "India Portfolio", period=lookback)
    us_series = build_cost_basis_series(us_builder, "US Portfolio", period=lookback)
    if ind_series is not None:
        series_list.append(ind_series)
    if us_series is not None:
        series_list.append(us_series)

    if series_list:
        chart_df = pd.concat(series_list, axis=1).reset_index().rename(columns={"index":"Date"})
        long_df = chart_df.melt(id_vars="Date", var_name="Series", value_name="Return")
        ch = alt.Chart(long_df).mark_line().encode(
            x="Date:T",
            y=alt.Y("Return:Q", axis=alt.Axis(format="%")),
            color="Series:N",
            tooltip=["Date:T","Series:N", alt.Tooltip("Return:Q", format=".2%")]
        ).properties(title=f"Performance ({lookback})")
        st.altair_chart(ch, use_container_width=True)
    else:
        st.info("No benchmark or portfolio series available for plotting.")

# ------------------ Expected ROI calculation and Target ROI planner ------------------
st.subheader("Target ROI Planner & Suggested Allocation")

# Compute expected returns per segment:
# For stocks: use india_growth and us_growth (user sliders)
# For bonds/fds: use average interest rates from uploaded data (or default conservative values)

def mean_interest_rate(df_asset):
    if df_asset is None or df_asset.empty or "InterestRate" not in df_asset.columns:
        return np.nan
    return float(df_asset["InterestRate"].dropna().astype(float).mean())

bonds_inr_rate = mean_interest_rate(bonds_inr_df)
bonds_usd_rate = mean_interest_rate(bonds_usd_df)
fds_inr_rate = mean_interest_rate(fds_inr_df)
fds_usd_rate = mean_interest_rate(fds_usd_df)

# defaults if NaN
bonds_inr_rate = bonds_inr_rate if pd.notna(bonds_inr_rate) else 6.0
bonds_usd_rate = bonds_usd_rate if pd.notna(bonds_usd_rate) else 3.0
fds_inr_rate = fds_inr_rate if pd.notna(fds_inr_rate) else 5.0
fds_usd_rate = fds_usd_rate if pd.notna(fds_usd_rate) else 2.5

# Convert all segment expected returns to a common basis (annual %)
# For INR assets, to express in USD terms we subtract INR depreciation
dep = inr_depr / 100.0
# India equity expected annual return (local)
r_india_local = india_growth / 100.0
# India equity expected in USD terms (approx): r_india_local - dep
r_india_usd_equiv = r_india_local - dep

r_us_local = us_growth / 100.0  # USD assets unaffected by INR depreciation

# For bonds/fds: convert INR rates to USD equivalent by subtracting depreciation
r_bonds_inr_usd_equiv = (bonds_inr_rate/100.0) - dep
r_fds_inr_usd_equiv = (fds_inr_rate/100.0) - dep
r_bonds_usd = bonds_usd_rate/100.0
r_fds_usd = fds_usd_rate/100.0

# Current amounts (from segments)
A_us_stock_usd = segments["us_stock_usd"]
A_ind_stock_usd = segments["india_stock_usd"]
A_us_bonds_usd = segments["bonds_usd_usd"]
A_ind_bonds_usd = segments["bonds_inr_usd"]
A_us_fds_usd = segments["fds_usd_usd"]
A_ind_fds_usd = segments["fds_inr_usd"]

total_usd = segments["total_usd"] if segments["total_usd"] else 0.0

# Avoid division by zero
if total_usd <= 0:
    st.warning("Total portfolio value is zero — upload portfolios to compute Target ROI suggestions.")
else:
    # Current weighted expected ROI (in USD terms) using current allocation
    # Use USD-equivalent returns for INR assets
    weights = {
        "us_stock": A_us_stock_usd/total_usd,
        "ind_stock": A_ind_stock_usd/total_usd,
        "us_bonds": A_us_bonds_usd/total_usd,
        "ind_bonds": A_ind_bonds_usd/total_usd,
        "us_fds": A_us_fds_usd/total_usd,
        "ind_fds": A_ind_fds_usd/total_usd
    }
    expected_current_roi = (
        weights["us_stock"] * r_us_local +
        weights["ind_stock"] * r_india_usd_equiv +
        weights["us_bonds"] * r_bonds_usd +
        weights["ind_bonds"] * r_bonds_inr_usd_equiv +
        weights["us_fds"] * r_fds_usd +
        weights["ind_fds"] * r_fds_inr_usd_equiv
    )

    st.write(f"Current expected (USD-equivalent) ROI from allocation: **{expected_current_roi*100:.2f}%** per year")
    st.write(f"Target ROI (user): **{target_roi:.2f}%** per year")

    # Simplified reallocation suggestion:
    # We'll treat all equity segments combined vs all fixed-income segments combined.
    # Solve for required equities weight w_e so that:
    # w_e * r_equities + (1 - w_e) * r_fixed = tR
    # Where r_equities = weighted average return of equity segments (using current split between India & US equity)
    # and r_fixed = weighted average return of fixed-income segments.

    # Compute current equities and fixed amounts and returns
    equities_usd = A_us_stock_usd + A_ind_stock_usd
    fixed_usd = A_us_bonds_usd + A_ind_bonds_usd + A_us_fds_usd + A_ind_fds_usd

    # If no fixed or no equities, handle
    r_equities = 0.0
    if equities_usd > 0:
        r_equities = (
            (A_us_stock_usd * r_us_local) + (A_ind_stock_usd * r_india_usd_equiv)
        ) / equities_usd

    r_fixed = 0.0
    if fixed_usd > 0:
        r_fixed = (
            (A_us_bonds_usd * r_bonds_usd) + (A_ind_bonds_usd * r_bonds_inr_usd_equiv) +
            (A_us_fds_usd * r_fds_usd) + (A_ind_fds_usd * r_fds_inr_usd_equiv)
        ) / fixed_usd

    tR = target_roi / 100.0
    # Solve for w_e
    suggested_equities_weight = None
    if (r_equities - r_fixed) == 0:
        suggested_equities_weight = np.nan  # cannot solve uniquely
    else:
        w_e = (tR - r_fixed) / (r_equities - r_fixed)
        suggested_equities_weight = float(np.clip(w_e, 0.0, 1.0))

    # Suggested allocation amounts:
    suggested_equities_amount_usd = suggested_equities_weight * total_usd if pd.notna(suggested_equities_weight) else np.nan
    suggested_fixed_amount_usd = (1.0 - suggested_equities_weight) * total_usd if pd.notna(suggested_equities_weight) else np.nan

    # Within equities: split India/US proportional to current equities composition (if equities present)
    if equities_usd > 0 and not np.isnan(suggested_equities_amount_usd):
        prop_ind = A_ind_stock_usd / equities_usd
        prop_us = A_us_stock_usd / equities_usd
        sug_ind_amount_usd = suggested_equities_amount_usd * prop_ind
        sug_us_amount_usd = suggested_equities_amount_usd * prop_us
    else:
        sug_ind_amount_usd = 0.0
        sug_us_amount_usd = 0.0

    # Within fixed: split across bonds/fds & INR/USD proportionally to current fixed composition
    if fixed_usd > 0 and not np.isnan(suggested_fixed_amount_usd):
        props = {
            "us_bonds": A_us_bonds_usd / fixed_usd,
            "ind_bonds": A_ind_bonds_usd / fixed_usd,
            "us_fds": A_us_fds_usd / fixed_usd,
            "ind_fds": A_ind_fds_usd / fixed_usd
        }
        sug_us_bonds = suggested_fixed_amount_usd * props["us_bonds"]
        sug_ind_bonds = suggested_fixed_amount_usd * props["ind_bonds"]
        sug_us_fds = suggested_fixed_amount_usd * props["us_fds"]
        sug_ind_fds = suggested_fixed_amount_usd * props["ind_fds"]
    else:
        sug_us_bonds = sug_ind_bonds = sug_us_fds = sug_ind_fds = 0.0

    # Build suggestion table in USD and INR
    suggest_rows = [
        {"Segment":"USD Stocks", "Current USD": A_us_stock_usd, "Suggested USD": sug_us_amount_usd},
        {"Segment":"INR Stocks", "Current USD": A_ind_stock_usd, "Suggested USD": sug_ind_amount_usd},
        {"Segment":"USD Bonds", "Current USD": A_us_bonds_usd, "Suggested USD": sug_us_bonds},
        {"Segment":"INR Bonds", "Current USD": A_ind_bonds_usd, "Suggested USD": sug_ind_bonds},
        {"Segment":"USD FDs", "Current USD": A_us_fds_usd, "Suggested USD": sug_us_fds},
        {"Segment":"INR FDs", "Current USD": A_ind_fds_usd, "Suggested USD": sug_ind_fds}
    ]
    suggest_df = pd.DataFrame(suggest_rows)
    suggest_df["Current INR"] = (suggest_df["Current USD"] * fx_rate).round(2)
    suggest_df["Suggested INR"] = (suggest_df["Suggested USD"] * fx_rate).round(2)
    suggest_df["Current USD"] = suggest_df["Current USD"].round(4)
    suggest_df["Suggested USD"] = suggest_df["Suggested USD"].round(4)
    # percent of total suggested
    if not np.isnan(suggested_equities_amount_usd):
        total_suggested = suggested_equities_amount_usd + suggested_fixed_amount_usd
    else:
        total_suggested = total_usd
    suggest_df["Suggested %"] = suggest_df["Suggested USD"].apply(lambda x: (x/total_suggested*100) if total_suggested>0 else 0.0).round(2)

    st.write("Suggested reallocation to achieve target ROI (USD-equivalent view):")
    st.dataframe(suggest_df[["Segment","Current USD","Current INR","Suggested USD","Suggested INR","Suggested %"]], use_container_width=True)

    # Pie chart for suggested allocation (INR amounts)
    pie_sugg = pd.DataFrame({
        "Segment": suggest_df["Segment"],
        "Suggested INR": suggest_df["Suggested INR"]
    })
    if pie_sugg["Suggested INR"].sum() > 0:
        p = alt.Chart(pie_sugg).mark_arc(innerRadius=30).encode(
            theta=alt.Theta(field="Suggested INR", type="quantitative"),
            color=alt.Color(field="Segment", type="nominal"),
            tooltip=["Segment", alt.Tooltip("Suggested INR", format=".2f")]
        ).properties(title="Suggested Allocation (INR amounts)")
        st.altair_chart(p, use_container_width=True)
    else:
        st.info("No suggested allocation (either target equals current or insufficient data).")

# ------------------ Exports: CSV/Excel ------------------
st.subheader("Export processed data")
# Stocks processed export
if not india_hold.empty:
    st.download_button("Download India Holdings CSV", data=india_hold.to_csv(index=False), file_name="india_holdings_processed.csv", mime="text/csv")
if not us_hold.empty:
    st.download_button("Download US Holdings CSV", data=us_hold.to_csv(index=False), file_name="us_holdings_processed.csv", mime="text/csv")

# Bonds & FDs CSV download (combined)
if not bonds_inr_df.empty:
    st.download_button("Download Bonds INR CSV", data=bonds_inr_df.to_csv(index=False), file_name="bonds_inr_processed.csv", mime="text/csv")
if not bonds_usd_df.empty:
    st.download_button("Download Bonds USD CSV", data=bonds_usd_df.to_csv(index=False), file_name="bonds_usd_processed.csv", mime="text/csv")
if not fds_inr_df.empty:
    st.download_button("Download FDs INR CSV", data=fds_inr_df.to_csv(index=False), file_name="fds_inr_processed.csv", mime="text/csv")
if not fds_usd_df.empty:
    st.download_button("Download FDs USD CSV", data=fds_usd_df.to_csv(index=False), file_name="fds_usd_processed.csv", mime="text/csv")

# Combined Excel export
if (not india_hold.empty) or (not us_hold.empty):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        if not india_hold.empty: india_hold.to_excel(writer, sheet_name="IndiaStocks", index=False)
        if not us_hold.empty: us_hold.to_excel(writer, sheet_name="USStocks", index=False)
        if not bonds_inr_df.empty: bonds_inr_df.to_excel(writer, sheet_name="Bonds_INR", index=False)
        if not bonds_usd_df.empty: bonds_usd_df.to_excel(writer, sheet_name="Bonds_USD", index=False)
        if not fds_inr_df.empty: fds_inr_df.to_excel(writer, sheet_name="FDs_INR", index=False)
        if not fds_usd_df.empty: fds_usd_df.to_excel(writer, sheet_name="FDs_USD", index=False)
    st.sidebar.download_button("Download All (Excel)", data=out.getvalue(), file_name="portfolio_full_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Notes: INR depreciation slider adjusts USD-equivalent returns for INR assets (approximation). Suggested allocation is a simplified rebalancing between equities vs fixed-income, then distributed proportionally across subsegments.")
