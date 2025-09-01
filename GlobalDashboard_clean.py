# -*- coding: utf-8 -*-
"""
GlobalDashboard_final.py
Full Streamlit app: stocks (INR+USD), bonds & FDs (INR+USD), allocation, ROI planner,
benchmarks (1mo/3mo/6mo/1y/2y), INR depreciation, CSV imports/exports, safe yfinance downloads.
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

# ---------------- Sidebar: uploads & settings ----------------
st.sidebar.header("Uploads & Settings")
india_file = st.sidebar.file_uploader("Upload Indian Portfolio CSV (Ticker,Quantity,AvgCost) — no .NS", type=["csv"])
us_file = st.sidebar.file_uploader("Upload US Portfolio CSV (Ticker,Quantity,AvgCost)", type=["csv"])

bonds_inr_file = st.sidebar.file_uploader("Upload Bonds (INR) CSV (Principal,Date,InterestRate%)", type=["csv"])
bonds_usd_file = st.sidebar.file_uploader("Upload Bonds (USD) CSV (Principal,Date,InterestRate%)", type=["csv"])
fds_inr_file = st.sidebar.file_uploader("Upload FDs (INR) CSV (Principal,Date,InterestRate%)", type=["csv"])
fds_usd_file = st.sidebar.file_uploader("Upload FDs (USD) CSV (Principal,Date,InterestRate%)", type=["csv"])

display_currency = st.sidebar.radio("Display currency", ["INR", "USD"], index=0)
lookback = st.sidebar.selectbox("Performance lookback period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
show_chart = st.sidebar.checkbox("Show Benchmarks & Portfolio Chart", value=True)

st.sidebar.subheader("Projection Inputs")
india_growth = st.sidebar.slider("India stocks growth rate (%)", -50.0, 50.0, 12.0, 0.1)
us_growth = st.sidebar.slider("US stocks growth rate (%)", -50.0, 50.0, 8.0, 0.1)
target_roi = st.sidebar.slider("Target overall ROI (%)", -50.0, 50.0, 10.0, 0.1)
inr_depr = st.sidebar.slider("INR depreciation vs USD (% per year)", 0.0, 10.0, 3.0, 0.1)

# optional auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh
    _ = st_autorefresh(interval=5 * 60 * 1000, key="autorefresh_final")
except Exception:
    pass

# ---------------- Utilities: safe_yf ----------------
@st.cache_data(ttl=300)
def safe_download(tickers, period="1y", interval="1d", group_by="ticker", threads=True):
    """Safe wrapper for yf.download that returns DataFrame or empty DataFrame."""
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
    """Return latest close for a list of symbols as a dict."""
    if not symbols:
        return {}
    out = {s: np.nan for s in symbols}
    data = safe_download(symbols, period="5d", interval="1d", group_by="ticker", threads=True)
    def extract_close(d, sym):
        try:
            if isinstance(d, pd.DataFrame) and isinstance(d.columns, pd.MultiIndex):
                ser = d[sym]["Close"].dropna()
                if not ser.empty:
                    return float(ser.iloc[-1])
            if isinstance(d, pd.DataFrame) and "Close" in d.columns:
                ser = d["Close"].dropna()
                if not ser.empty:
                    return float(ser.iloc[-1])
        except Exception:
            pass
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

@st.cache_data(ttl=600)
def get_fx_rate():
    """USD per INR (we keep USDINR as INR per USD)."""
    try:
        finf = yf.Ticker("USDINR=X").fast_info
        if finf and finf.get("lastPrice") is not None:
            return float(finf["lastPrice"])
    except Exception:
        pass
    d = safe_download("USDINR=X", period="1d", interval="5m")
    try:
        if not d.empty and "Close" in d.columns:
            return float(d["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return 83.0

fx_rate = get_fx_rate()  # INR per 1 USD

# ---------------- Loaders ----------------
def load_stock_csv(f, is_india=True):
    if f is None:
        return pd.DataFrame()
    df = pd.read_csv(f)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df = df.rename(columns={"Ticker":"ticker","Quantity":"shares","AvgCost":"avg_cost"})
    df["shares"] = pd.to_numeric(df.get("shares"), errors="coerce")
    df["avg_cost"] = pd.to_numeric(df.get("avg_cost"), errors="coerce")
    if is_india:
        df["ticker"] = df["ticker"].astype(str).apply(lambda x: x if str(x).endswith(".NS") else f"{x}.NS")
        df["currency"] = "INR"
    else:
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["currency"] = "USD"
    return df

def load_asset_csv(f, currency="INR"):
    if f is None:
        return pd.DataFrame()
    df = pd.read_csv(f)
    # normalize basic columns
    cols = {c: c.strip().title() for c in df.columns}
    df = df.rename(columns=cols)
    if "Principal" not in df.columns:
        df["Principal"] = pd.to_numeric(df.iloc[:,0], errors="coerce")
    else:
        df["Principal"] = pd.to_numeric(df["Principal"], errors="coerce")
    # interest col detection
    ir_col = None
    for c in df.columns:
        if "interest" in c.lower(): ir_col = c
    if ir_col:
        df["InterestRate"] = pd.to_numeric(df[ir_col], errors="coerce")
    else:
        df["InterestRate"] = np.nan
    df["Currency"] = currency
    return df

# ---------------- Process Stocks ----------------
def process_stocks(df):
    """Return DataFrame with both INR/USD cost & market values."""
    if df is None or df.empty:
        return pd.DataFrame()
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
            # native INR
            mv_inr = mv_native
            mv_usd = (mv_native / fx_rate) if pd.notna(mv_native) else np.nan
            cost_inr = cost_native
            cost_usd = (cost_native / fx_rate) if pd.notna(cost_native) else np.nan
            avg_native = avg_cost
        else:
            # native USD
            mv_usd = mv_native
            mv_inr = (mv_native * fx_rate) if pd.notna(mv_native) else np.nan
            cost_usd = cost_native
            cost_inr = (cost_native * fx_rate) if pd.notna(cost_native) else np.nan
            avg_native = avg_cost

        rows.append({
            "Ticker": ticker,
            "Shares": shares,
            "Avg Cost (native)": avg_native,
            "Cost INR": cost_inr,
            "Cost USD": cost_usd,
            "Market Value INR": mv_inr,
            "Market Value USD": mv_usd
        })
    return pd.DataFrame(rows)

# ---------------- Load uploaded data ----------------
india_raw = load_stock_csv(india_file, is_india=True)
us_raw = load_stock_csv(us_file, is_india=False)

india_hold = process_stocks(india_raw) if not india_raw.empty else pd.DataFrame()
us_hold = process_stocks(us_raw) if not us_raw.empty else pd.DataFrame()

bonds_inr_df = load_asset_csv(bonds_inr_file, "INR")
bonds_usd_df = load_asset_csv(bonds_usd_file, "USD")
fds_inr_df = load_asset_csv(fds_inr_file, "INR")
fds_usd_df = load_asset_csv(fds_usd_file, "USD")

# Manual entry for bonds/fds (optional)
def manual_add_asset(key, label, currency):
    if key not in st.session_state:
        st.session_state[key] = []
    with st.form(f"form_{key}", clear_on_submit=True):
        p = st.number_input(f"{label} Principal ({currency})", min_value=0.0, value=100000.0)
        d = st.date_input(f"{label} Date", value=date.today())
        r = st.number_input(f"{label} Interest Rate (%)", min_value=0.0, value=5.0)
        submit = st.form_submit_button("Add")
    if submit:
        st.session_state[key].append({"Principal": float(p), "Date": str(d), "InterestRate": float(r), "Currency": currency})

st.sidebar.subheader("Manual Bond/FD entry (optional)")
manual_add_asset("bonds_inr_manual", "Bond (INR)", "INR")
manual_add_asset("bonds_usd_manual", "Bond (USD)", "USD")
manual_add_asset("fds_inr_manual", "FD (INR)", "INR")
manual_add_asset("fds_usd_manual", "FD (USD)", "USD")

# Merge manual entries into dfs
bonds_inr_manual = pd.DataFrame(st.session_state.get("bonds_inr_manual", []))
bonds_usd_manual = pd.DataFrame(st.session_state.get("bonds_usd_manual", []))
fds_inr_manual = pd.DataFrame(st.session_state.get("fds_inr_manual", []))
fds_usd_manual = pd.DataFrame(st.session_state.get("fds_usd_manual", []))

if not bonds_inr_manual.empty:
    bonds_inr_df = pd.concat([bonds_inr_df, bonds_inr_manual], ignore_index=True) if not bonds_inr_df.empty else bonds_inr_manual.copy()
if not bonds_usd_manual.empty:
    bonds_usd_df = pd.concat([bonds_usd_df, bonds_usd_manual], ignore_index=True) if not bonds_usd_df.empty else bonds_usd_manual.copy()
if not fds_inr_manual.empty:
    fds_inr_df = pd.concat([fds_inr_df, fds_inr_manual], ignore_index=True) if not fds_inr_df.empty else fds_inr_manual.copy()
if not fds_usd_manual.empty:
    fds_usd_df = pd.concat([fds_usd_df, fds_usd_manual], ignore_index=True) if not fds_usd_df.empty else fds_usd_manual.copy()

# ---------------- Portfolio Summary (per stock table + totals) ----------------
st.subheader("Portfolio Summaries")

def show_holdings_table(df_hold, title, native_currency_label):
    if df_hold is None or df_hold.empty:
        st.write(f"{title}: No data.")
        return
    st.write(f"**{title}**")
    display = df_hold.copy()
    # show selected columns and round
    display["Market Value INR"] = display["Market Value INR"].round(2)
    display["Market Value USD"] = display["Market Value USD"].round(4)
    display["Cost INR"] = display["Cost INR"].round(2)
    display["Cost USD"] = display["Cost USD"].round(4)
    st.dataframe(display[["Ticker","Shares","Avg Cost (native)","Cost INR","Cost USD","Market Value INR","Market Value USD"]], use_container_width=True)
    total_cost_inr = display["Cost INR"].sum()
    total_mv_inr = display["Market Value INR"].sum()
    total_pl_inr = total_mv_inr - total_cost_inr
    st.write(f"Total Cost (INR): {total_cost_inr:,.2f} | Market Value (INR): {total_mv_inr:,.2f} | Unrealized P/L (INR): {total_pl_inr:,.2f}")

show_holdings_table(india_hold, "India Holdings", "INR")
show_holdings_table(us_hold, "US Holdings", "USD")

# ---------------- Allocation (6 segments) ----------------
def segment_sums():
    india_stock_inr = india_hold["Market Value INR"].sum() if not india_hold.empty else 0.0
    india_stock_usd = (india_stock_inr / fx_rate) if fx_rate and india_stock_inr else 0.0
    us_stock_usd = us_hold["Market Value USD"].sum() if not us_hold.empty else 0.0
    us_stock_inr = us_stock_usd * fx_rate if fx_rate and us_stock_usd else 0.0

    bonds_inr_total = bonds_inr_df["Principal"].sum() if not bonds_inr_df.empty else 0.0
    bonds_usd_total = bonds_usd_df["Principal"].sum() if not bonds_usd_df.empty else 0.0
    fds_inr_total = fds_inr_df["Principal"].sum() if not fds_inr_df.empty else 0.0
    fds_usd_total = fds_usd_df["Principal"].sum() if not fds_usd_df.empty else 0.0

    return {
        "USD Stocks": {"USD": us_stock_usd, "INR": us_stock_inr},
        "INR Stocks": {"INR": india_stock_inr, "USD": india_stock_usd},
        "USD Bonds": {"USD": bonds_usd_total, "INR": bonds_usd_total * fx_rate},
        "INR Bonds": {"INR": bonds_inr_total, "USD": bonds_inr_total / fx_rate if fx_rate else 0.0},
        "USD FDs": {"USD": fds_usd_total, "INR": fds_usd_total * fx_rate},
        "INR FDs": {"INR": fds_inr_total, "USD": fds_inr_total / fx_rate if fx_rate else 0.0}
    }

segments = segment_sums()
alloc_rows = []
for seg, vals in segments.items():
    amt_inr = float(vals.get("INR", 0.0))
    amt_usd = float(vals.get("USD", 0.0))
    alloc_rows.append({"Segment": seg, "Amount INR": round(amt_inr,2), "Amount USD": round(amt_usd,4)})
alloc_df = pd.DataFrame(alloc_rows)
total_inr = alloc_df["Amount INR"].sum()
alloc_df["Percent (%)"] = alloc_df["Amount INR"].apply(lambda x: round((x/total_inr*100) if total_inr>0 else 0.0,2))
st.subheader("Allocation Split — INR & USD amounts and %")
st.dataframe(alloc_df, use_container_width=True)

pie = alt.Chart(alloc_df).mark_arc(innerRadius=50).encode(
    theta=alt.Theta(field="Amount INR", type="quantitative"),
    color=alt.Color(field="Segment", type="nominal"),
    tooltip=["Segment", alt.Tooltip("Amount INR", format=",.2f"), alt.Tooltip("Amount USD", format=",.4f"), "Percent (%)"]
).properties(title="Allocation by INR amount")
st.altair_chart(pie, use_container_width=True)

# ---------------- Benchmarks + portfolio performance chart ----------------
if show_chart:
    st.subheader("Benchmark & Portfolio Performance (cost-basis)")
    def bench_series(sym, period=lookback):
        df = safe_download(sym, period=period, interval="1d")
        if df.empty: return pd.Series(dtype=float)
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

    series_list = []
    for name, sym in {"S&P 500":"^GSPC","NASDAQ":"^IXIC","NIFTY50":"^NSEI"}.items():
        s = bench_series(sym, period=lookback)
        if not s.empty:
            s.name = name
            series_list.append(s)

    # cost-basis portfolio series (use avg cost native, weight by cost INR)
    def build_cost_basis(df_hold, label, period=lookback):
        if df_hold is None or df_hold.empty: return None
        # We need avg native and cost in INR to weight
        parts = []
        total_cost_inr = df_hold["Cost INR"].sum()
        if not pd.notna(total_cost_inr) or total_cost_inr <= 0: return None
        for _, r in df_hold.iterrows():
            ticker = r["Ticker"]
            avg_native = r["Avg Cost (native)"]
            # avg_native may be in INR for .NS or USD for US tickers; but we normalize with cost INR weighting
            cost_inr = r["Cost INR"]
            weight = (cost_inr / total_cost_inr) if pd.notna(cost_inr) and total_cost_inr>0 else 0.0
            if not (pd.notna(avg_native) and avg_native>0 and weight>0):
                continue
            hist = safe_download(ticker, period=period, interval="1d")
            if hist.empty: continue
            try:
                close = hist["Close"] if "Close" in hist.columns else hist[ticker]["Close"]
            except Exception:
                continue
            contrib = (close / float(avg_native) - 1.0) * float(weight)
            contrib.name = ticker
            parts.append(contrib)
        if not parts: return None
        df_aligned = pd.concat(parts, axis=1).fillna(0.0)
        series = df_aligned.sum(axis=1)
        series.name = label
        return series

    # prepare small df with Avg Cost native & Cost INR for builder
    def prepare_builder_df(hold_df):
        if hold_df is None or hold_df.empty: return pd.DataFrame()
        out = pd.DataFrame()
        out["Ticker"] = hold_df["Ticker"]
        out["Avg Cost (native)"] = hold_df["Avg Cost (native)"]
        out["Cost INR"] = hold_df["Cost INR"]
        return out

    # Note: our process_stocks created Cost INR/Cost USD; ensure presence
    # If missing, attempt to calculate from Market Value and last price.
    # For safety, fill missing Cost INR as Market Value INR (approx)
    if "Cost INR" not in india_hold.columns and "Market Value INR" in india_hold.columns:
        india_hold["Cost INR"] = india_hold["Market Value INR"]
    if "Cost INR" not in us_hold.columns and "Market Value USD" in us_hold.columns:
        us_hold["Cost INR"] = us_hold["Market Value USD"] * fx_rate if fx_rate else us_hold["Market Value USD"]

    india_builder = prepare_builder_df(india_hold)
    us_builder = prepare_builder_df(us_hold)

    ind_series = build_cost_basis(india_builder, "India Portfolio", period=lookback)
    us_series = build_cost_basis(us_builder, "US Portfolio", period=lookback)
    if ind_series is not None: series_list.append(ind_series)
    if us_series is not None: series_list.append(us_series)

    if series_list:
        chart_df = pd.concat(series_list, axis=1).reset_index().rename(columns={"index":"Date"})
        long_df = chart_df.melt(id_vars="Date", var_name="Series", value_name="Return")
        chart = alt.Chart(long_df).mark_line().encode(
            x="Date:T",
            y=alt.Y("Return:Q", axis=alt.Axis(format="%")),
            color="Series:N",
            tooltip=["Date:T","Series:N", alt.Tooltip("Return:Q", format=".2%")]
        ).properties(title=f"Performance ({lookback})")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Not enough data to plot benchmarks/portfolios for selected lookback.")

# ------------------ Consistency check (summary vs chart) ----------------
def check_alignment(proc_df, series, label):
    if proc_df is None or proc_df.empty or series is None or series.empty: return
    tot_cost = proc_df["Cost INR"].sum()
    tot_mv = proc_df["Market Value INR"].sum()
    if not (pd.notna(tot_cost) and tot_cost>0 and pd.notna(tot_mv)): return
    summary_ret = (tot_mv - tot_cost) / tot_cost
    chart_ret = series.dropna().iloc[-1]
    st.caption(f"🔎 {label}: Summary return = {summary_ret:.2%}, Chart return = {chart_ret:.2%}, Δ = {(summary_ret - chart_ret):.2%}")

# Build simple proc_df containing Cost INR and Market Value INR if not present
# (our process_stocks already created Cost INR & Market Value INR)
if not india_hold.empty and (ind_series is not None):
    check_alignment(india_hold, ind_series, "India Portfolio")
if not us_hold.empty and (us_series is not None):
    check_alignment(us_hold, us_series, "US Portfolio")

# ------------------ ROI Planner & Suggested Allocation ----------------
st.subheader("Target ROI Planner & Suggested Allocation")

# compute returns for each segment (USD-equivalent)
dep = inr_depr / 100.0
r_india_local = india_growth / 100.0
r_india_usd_equiv = r_india_local - dep
r_us_local = us_growth / 100.0

# bonds/fds mean rates
def mean_rate(df_asset):
    if df_asset is None or df_asset.empty or "InterestRate" not in df_asset.columns:
        return np.nan
    return float(pd.to_numeric(df_asset["InterestRate"], errors="coerce").dropna().mean())

r_bonds_inr = mean_rate(bonds_inr_df)/100.0 if mean_rate(bonds_inr_df) is not np.nan else 0.06
r_bonds_usd = mean_rate(bonds_usd_df)/100.0 if mean_rate(bonds_usd_df) is not np.nan else 0.03
r_fds_inr = mean_rate(fds_inr_df)/100.0 if mean_rate(fds_inr_df) is not np.nan else 0.05
r_fds_usd = mean_rate(fds_usd_df)/100.0 if mean_rate(fds_usd_df) is not np.nan else 0.025

r_bonds_inr_usd_equiv = r_bonds_inr - dep
r_fds_inr_usd_equiv = r_fds_inr - dep

# Current amounts in USD
A_us_stock = segments["USD Stocks"]["USD"]
A_ind_stock = segments["INR Stocks"]["USD"]
A_us_bonds = segments["USD Bonds"]["USD"]
A_ind_bonds = segments["INR Bonds"]["USD"]
A_us_fds = segments["USD FDs"]["USD"]
A_ind_fds = segments["INR FDs"]["USD"]

total_usd = sum([A_us_stock, A_ind_stock, A_us_bonds, A_ind_bonds, A_us_fds, A_ind_fds])
if total_usd <= 0:
    st.warning("Total portfolio value is zero — upload data to get suggestions.")
else:
    # expected current ROI (USD-equivalent)
    expected_current = (
        (A_us_stock/total_usd)*r_us_local +
        (A_ind_stock/total_usd)*r_india_usd_equiv +
        (A_us_bonds/total_usd)*r_bonds_usd +
        (A_ind_bonds/total_usd)*r_bonds_inr_usd_equiv +
        (A_us_fds/total_usd)*r_fds_usd +
        (A_ind_fds/total_usd)*r_fds_inr_usd_equiv
    )
    st.write(f"Expected current (USD-equivalent) ROI: **{expected_current*100:.2f}%** per year. Target: **{target_roi:.2f}%**.")

    # Simplified approach: solve for equities weight vs fixed weight
    equities_usd = A_us_stock + A_ind_stock
    fixed_usd = A_us_bonds + A_ind_bonds + A_us_fds + A_ind_fds

    # average returns
    r_equities = ((A_us_stock * r_us_local) + (A_ind_stock * r_india_usd_equiv)) / equities_usd if equities_usd>0 else 0.0
    r_fixed = ((A_us_bonds * r_bonds_usd) + (A_ind_bonds * r_bonds_inr_usd_equiv) + (A_us_fds * r_fds_usd) + (A_ind_fds * r_fds_inr_usd_equiv)) / fixed_usd if fixed_usd>0 else 0.0

    tR = target_roi / 100.0
    if (r_equities - r_fixed) == 0:
        suggested_equities_weight = np.nan
    else:
        w_e = (tR - r_fixed) / (r_equities - r_fixed)
        suggested_equities_weight = float(np.clip(w_e, 0.0, 1.0))

    suggested_equities_amount = suggested_equities_weight * total_usd if not np.isnan(suggested_equities_weight) else np.nan
    suggested_fixed_amount = total_usd - suggested_equities_amount if not np.isnan(suggested_equities_amount) else np.nan

    # within equities split proportionally to existing equity composition
    prop_ind_equity = (A_ind_stock / equities_usd) if equities_usd>0 else 0.0
    prop_us_equity = (A_us_stock / equities_usd) if equities_usd>0 else 0.0

    sug_ind_equity_usd = suggested_equities_amount * prop_ind_equity if not np.isnan(suggested_equities_amount) else 0.0
    sug_us_equity_usd = suggested_equities_amount * prop_us_equity if not np.isnan(suggested_equities_amount) else 0.0

    # within fixed split proportionally to current fixed composition
    props_fixed = {}
    if fixed_usd>0:
        props_fixed["us_bonds"] = A_us_bonds / fixed_usd
        props_fixed["ind_bonds"] = A_ind_bonds / fixed_usd
        props_fixed["us_fds"] = A_us_fds / fixed_usd
        props_fixed["ind_fds"] = A_ind_fds / fixed_usd
    else:
        props_fixed = {"us_bonds":0,"ind_bonds":0,"us_fds":0,"ind_fds":0}

    sug_us_bonds = suggested_fixed_amount * props_fixed["us_bonds"] if not np.isnan(suggested_fixed_amount) else 0.0
    sug_ind_bonds = suggested_fixed_amount * props_fixed["ind_bonds"] if not np.isnan(suggested_fixed_amount) else 0.0
    sug_us_fds = suggested_fixed_amount * props_fixed["us_fds"] if not np.isnan(suggested_fixed_amount) else 0.0
    sug_ind_fds = suggested_fixed_amount * props_fixed["ind_fds"] if not np.isnan(suggested_fixed_amount) else 0.0

    # Build suggestion DataFrame
    suggest = pd.DataFrame([
        {"Segment":"USD Stocks", "Current USD": A_us_stock, "Suggested USD": sug_us_equity_usd},
        {"Segment":"INR Stocks", "Current USD": A_ind_stock, "Suggested USD": sug_ind_equity_usd},
        {"Segment":"USD Bonds", "Current USD": A_us_bonds, "Suggested USD": sug_us_bonds},
        {"Segment":"INR Bonds", "Current USD": A_ind_bonds, "Suggested USD": sug_ind_bonds},
        {"Segment":"USD FDs", "Current USD": A_us_fds, "Suggested USD": sug_us_fds},
        {"Segment":"INR FDs", "Current USD": A_ind_fds, "Suggested USD": sug_ind_fds},
    ])
    suggest["Current INR"] = (suggest["Current USD"] * fx_rate).round(2)
    suggest["Suggested INR"] = (suggest["Suggested USD"] * fx_rate).round(2)
    suggest["Current USD"] = suggest["Current USD"].round(4)
    suggest["Suggested USD"] = suggest["Suggested USD"].round(4)
    total_suggested_usd = suggest["Suggested USD"].sum() if suggest["Suggested USD"].sum() > 0 else total_usd
    suggest["Suggested %"] = suggest["Suggested USD"].apply(lambda x: round((x/total_suggested_usd*100) if total_suggested_usd>0 else 0.0,2))

    st.write("Suggested reallocation (USD & INR amounts + % of suggested total):")
    st.dataframe(suggest[["Segment","Current USD","Current INR","Suggested USD","Suggested INR","Suggested %"]], use_container_width=True)

    # suggested pie (INR)
    pie_sug = alt.Chart(suggest).mark_arc(innerRadius=30).encode(
        theta=alt.Theta(field="Suggested INR", type="quantitative"),
        color=alt.Color(field="Segment", type="nominal"),
        tooltip=["Segment", alt.Tooltip("Suggested INR", format=",.2f")]
    ).properties(title="Suggested Allocation (INR amounts)")
    st.altair_chart(pie_sug, use_container_width=True)

# ------------------ Exports ----------------
st.subheader("Export processed data")
if not india_hold.empty:
    st.download_button("Download India Holdings (processed CSV)", data=india_hold.to_csv(index=False), file_name="india_holdings_processed.csv", mime="text/csv")
if not us_hold.empty:
    st.download_button("Download US Holdings (processed CSV)", data=us_hold.to_csv(index=False), file_name="us_holdings_processed.csv", mime="text/csv")

if not bonds_inr_df.empty:
    st.download_button("Download Bonds INR CSV", data=bonds_inr_df.to_csv(index=False), file_name="bonds_inr.csv", mime="text/csv")
if not bonds_usd_df.empty:
    st.download_button("Download Bonds USD CSV", data=bonds_usd_df.to_csv(index=False), file_name="bonds_usd.csv", mime="text/csv")
if not fds_inr_df.empty:
    st.download_button("Download FDs INR CSV", data=fds_inr_df.to_csv(index=False), file_name="fds_inr.csv", mime="text/csv")
if not fds_usd_df.empty:
    st.download_button("Download FDs USD CSV", data=fds_usd_df.to_csv(index=False), file_name="fds_usd.csv", mime="text/csv")

# Combined Excel
if not (india_hold.empty and us_hold.empty):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        if not india_hold.empty: india_hold.to_excel(writer, sheet_name="IndiaStocks", index=False)
        if not us_hold.empty: us_hold.to_excel(writer, sheet_name="USStocks", index=False)
        if not bonds_inr_df.empty: bonds_inr_df.to_excel(writer, sheet_name="Bonds_INR", index=False)
        if not bonds_usd_df.empty: bonds_usd_df.to_excel(writer, sheet_name="Bonds_USD", index=False)
        if not fds_inr_df.empty: fds_inr_df.to_excel(writer, sheet_name="FDs_INR", index=False)
        if not fds_usd_df.empty: fds_usd_df.to_excel(writer, sheet_name="FDs_USD", index=False)
    st.sidebar.download_button("Download All (Excel)", data=out.getvalue(), file_name="portfolio_full_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Notes: INR depreciation slider approximates USD-equivalent returns for INR assets. Suggested allocation is a simplified rebalancing method (equities vs fixed income) and then proportional distribution.")
