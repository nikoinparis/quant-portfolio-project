import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

st.set_page_config(
    page_title="Quant Portfolio Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0b1020 0%, #111827 100%);
        color: #e5e7eb;
    }

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 1.6rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1450px;
    }

    .metric-card {
        background: rgba(17, 24, 39, 0.92);
        border: 1px solid rgba(99, 102, 241, 0.28);
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 0 0 1px rgba(79, 70, 229, 0.08), 0 10px 30px rgba(0, 0, 0, 0.25);
        min-height: 135px;
    }

    .section-card {
        background: rgba(17, 24, 39, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 20px;
        padding: 18px 20px 14px 20px;
        box-shadow: 0 12px 34px rgba(0, 0, 0, 0.18);
        margin-top: 0.8rem;
        margin-bottom: 1.2rem;
    }

    .small-label {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-bottom: 0.15rem;
    }

    .big-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }

    .title-text {
        font-size: 2.55rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: 0.01em;
        margin-top: 0.1rem;
        margin-bottom: 0.45rem;
        line-height: 1.08;
    }

    .subtitle-text {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
def safe_read_csv(path: Path):
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return None


def safe_read_csv_no_index(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


def compute_metrics(return_series: pd.Series) -> dict:
    if return_series is None:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan, "Calmar": np.nan}

    rs = return_series.dropna()
    if len(rs) == 0:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan, "Calmar": np.nan}

    cumulative = (1 + rs).cumprod()
    months = len(rs)
    cagr = cumulative.iloc[-1] ** (12 / months) - 1
    vol = rs.std() * np.sqrt(12)
    sharpe = cagr / vol if vol and not np.isnan(vol) else np.nan
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd and not np.isnan(max_dd) else np.nan

    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": max_dd, "Calmar": calmar}


def fmt_pct(x):
    return "—" if pd.isna(x) else f"{x:.2%}"


def fmt_num(x):
    return "—" if pd.isna(x) else f"{x:.2f}"


def draw_metric_card(label: str, value: str, sublabel: str = ""):
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='small-label'>{label}</div>
            <div class='big-value'>{value}</div>
            <div class='small-label'>{sublabel}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalize_returns(df):
    if isinstance(df, pd.Series):
        return (1 + df.fillna(0)).cumprod()
    return (1 + df.fillna(0)).cumprod()


# ---------- Load core files ----------
final_comparison_returns = safe_read_csv(DATA_DIR / "final_comparison_returns.csv")
final_comparison_metrics = safe_read_csv(DATA_DIR / "final_comparison_metrics.csv")
final_summary = safe_read_csv(DATA_DIR / "final_strategy_summary_table.csv")
monthly_returns = safe_read_csv(DATA_DIR / "monthly_returns.csv")
best_etf_weights = safe_read_csv(DATA_DIR / "best_etf_level_weights.csv")
walk_forward_selected = safe_read_csv_no_index(DATA_DIR / "walk_forward_selected_models.csv")

# ---------- Load shared strategy weight files ----------
shared_120_weights = safe_read_csv(DATA_DIR / "shared_120d_target_weights.csv")
shared_180_weights = safe_read_csv(DATA_DIR / "shared_180d_target_weights.csv")
shared_252_weights = safe_read_csv(DATA_DIR / "shared_252d_target_weights.csv")

# ---------- Strategy registry ----------
strategy_registry = {
    "Shared 120d": {"column": "Shared_120d", "type": "core"},
    "Shared 180d": {"column": "Shared_180d", "type": "core"},
    "Shared 252d": {"column": "Shared_252d", "type": "core"},
    "Best Static ETF-Level": {"column": "Best_Static_ETF_Level", "type": "advanced"},
}

available_strategies = []
if final_comparison_returns is not None:
    for name, meta in strategy_registry.items():
        if meta["column"] in final_comparison_returns.columns:
            available_strategies.append(name)

# ---------- Sidebar ----------
st.sidebar.markdown("## Strategy Controls")

if available_strategies:
    default_index = available_strategies.index("Shared 252d") if "Shared 252d" in available_strategies else 0
    selected_strategy = st.sidebar.selectbox("Strategy Engine", available_strategies, index=default_index)
else:
    selected_strategy = None
    st.sidebar.warning("No registered strategies were found in final_comparison_returns.csv")

benchmark_defaults = [
    c for c in ["SPY_BuyHold", "Portfolio_60_40", "DBMF", "KMLM", "RPAR"]
    if final_comparison_returns is not None and c in final_comparison_returns.columns
]

selected_benchmarks = st.sidebar.multiselect(
    "Benchmarks",
    options=list(final_comparison_returns.columns) if final_comparison_returns is not None else [],
    default=benchmark_defaults,
)

chart_mode = st.sidebar.radio(
    "Secondary chart",
    options=["Drawdown", "Rolling Volatility"],
    index=0,
)

# ---------- Header ----------
st.markdown("<div class='title-text'>Quant Portfolio Dashboard</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle-text'>Signal-driven tactical allocation dashboard with room for robust optimization engines later.</div>",
    unsafe_allow_html=True,
)

if final_comparison_returns is None:
    st.error("No processed strategy files found yet. Make sure final_comparison_returns.csv exists in data/processed/.")
    st.stop()

# ---------- Selected series ----------
selected_col = strategy_registry[selected_strategy]["column"] if selected_strategy else None

if selected_col is not None and selected_col in final_comparison_returns.columns:
    strategy_series = final_comparison_returns[selected_col]
    strategy_metrics = compute_metrics(strategy_series)
else:
    strategy_series = None
    strategy_metrics = {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan, "Calmar": np.nan}

current_value = normalize_returns(strategy_series).iloc[-1] if strategy_series is not None else np.nan
start_date = strategy_series.dropna().index.min().date() if strategy_series is not None and len(strategy_series.dropna()) > 0 else "—"
end_date = strategy_series.dropna().index.max().date() if strategy_series is not None and len(strategy_series.dropna()) > 0 else "—"

# ---------- Top metrics ----------
metric_cols = st.columns(5, gap="medium")

with metric_cols[0]:
    draw_metric_card("Selected Strategy", selected_strategy or "—", f"Backtest window: {start_date} → {end_date}")

with metric_cols[1]:
    draw_metric_card("Growth of $1", fmt_num(current_value), "Cumulative portfolio value")

with metric_cols[2]:
    draw_metric_card("CAGR", fmt_pct(strategy_metrics["CAGR"]), "Annualized return")

with metric_cols[3]:
    draw_metric_card("Sharpe", fmt_num(strategy_metrics["Sharpe"]), "Risk-adjusted return")

with metric_cols[4]:
    draw_metric_card("Max Drawdown", fmt_pct(strategy_metrics["MaxDD"]), "Worst peak-to-trough decline")

# ---------- Main comparison chart ----------
left, right = st.columns([1.65, 1], gap="large")

with left:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Performance Comparison")

    compare_cols = [selected_col] + selected_benchmarks
    compare_cols = [c for c in compare_cols if c is not None and c in final_comparison_returns.columns]

    comparison_df = normalize_returns(final_comparison_returns[compare_cols])
    st.line_chart(comparison_df, height=400)
    st.caption("Normalized cumulative performance of the selected strategy against chosen benchmarks.")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Strategy Metrics")

    summary_cols = [selected_col] + selected_benchmarks
    summary_cols = [c for c in summary_cols if c is not None and c in final_comparison_returns.columns]

    metrics_table = pd.DataFrame({col: compute_metrics(final_comparison_returns[col]) for col in summary_cols}).T
    st.dataframe(
        metrics_table.style.format({
            "CAGR": "{:.2%}",
            "Vol": "{:.2%}",
            "Sharpe": "{:.2f}",
            "MaxDD": "{:.2%}",
            "Calmar": "{:.2f}",
        }),
        use_container_width=True,
        height=400,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Secondary row ----------
left2, right2 = st.columns([1.15, 1], gap="large")

with left2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Current / Latest Weights")

    weight_map = {
        "Shared 120d": shared_120_weights,
        "Shared 180d": shared_180_weights,
        "Shared 252d": shared_252_weights,
        "Best Static ETF-Level": best_etf_weights,
    }

    selected_weight_df = weight_map.get(selected_strategy)

    if selected_weight_df is not None:
        latest_weights = selected_weight_df.iloc[-1].dropna()
        latest_weights = latest_weights[latest_weights > 0].sort_values(ascending=False)

        if len(latest_weights) > 0:
            latest_df = latest_weights.rename("Weight").to_frame()
            st.bar_chart(latest_df, height=300)
            st.dataframe(
                latest_df.style.format({"Weight": "{:.2%}"}),
                use_container_width=True,
                height=250,
            )
        else:
            st.info("No positive weights were found in the latest saved row for this strategy.")
    else:
        st.warning("No saved weight file was found for this strategy yet.")

    st.markdown("</div>", unsafe_allow_html=True)

with right2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)

    if chart_mode == "Drawdown":
        st.subheader("Drawdown")
        dd = comparison_df.div(comparison_df.cummax()) - 1
        st.line_chart(dd, height=300)
        st.caption("Peak-to-trough drawdown for the selected strategy and benchmarks.")
    else:
        st.subheader("Rolling Volatility")
        roll_vol = final_comparison_returns[compare_cols].rolling(12).std() * np.sqrt(12)
        st.line_chart(roll_vol, height=300)
        st.caption("12-month rolling annualized volatility.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Diagnostics ----------
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("Research / Diagnostics")

research_tab1, research_tab2, research_tab3 = st.tabs(
    ["Final Summary", "Return Correlations", "Walk-Forward Selections"]
)

with research_tab1:
    if final_summary is not None:
        st.dataframe(
            final_summary.style.format({
                "CAGR": "{:.2%}",
                "Annual Vol": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Max Drawdown": "{:.2%}",
                "Calmar": "{:.2f}",
            }),
            use_container_width=True,
        )
    elif final_comparison_metrics is not None:
        st.dataframe(
            final_comparison_metrics.style.format({
                "CAGR": "{:.2%}",
                "Annual Vol": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Max Drawdown": "{:.2%}",
                "Calmar": "{:.2f}",
            }),
            use_container_width=True,
        )
    else:
        st.info("No final strategy summary table found yet.")

with research_tab2:
    corr_cols = [selected_col] + selected_benchmarks
    corr_cols = [c for c in corr_cols if c is not None and c in final_comparison_returns.columns]
    corr = final_comparison_returns[corr_cols].corr()
    st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)

with research_tab3:
    if walk_forward_selected is not None:
        st.dataframe(walk_forward_selected, use_container_width=True)
    else:
        st.info("No walk-forward selection log found yet.")

st.markdown("</div>", unsafe_allow_html=True)

# ---------- Footer ----------
st.caption(
    "Next steps: add live signal display, benchmark download caching, transaction-cost analysis, and broker/paper-trading integration."
)