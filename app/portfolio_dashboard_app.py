import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DATA_DIR = BASE_DIR / "data" / "processed" / "app_data"

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
        padding-top: 1.65rem;
        padding-bottom: 1.5rem;
        padding-left: 1.8rem;
        padding-right: 1.8rem;
        max-width: 1520px;
    }

    .metric-card {
        background: rgba(17, 24, 39, 0.92);
        border: 1px solid rgba(99, 102, 241, 0.28);
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 0 0 1px rgba(79, 70, 229, 0.08), 0 10px 30px rgba(0, 0, 0, 0.25);
        min-height: 128px;
    }

    .section-card {
        background: rgba(17, 24, 39, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 20px;
        padding: 18px 20px 14px 20px;
        box-shadow: 0 12px 34px rgba(0, 0, 0, 0.18);
        margin-top: 0.65rem;
        margin-bottom: 1rem;
    }

    .small-label {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-bottom: 0.15rem;
    }

    .big-value {
        color: #f8fafc;
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }

    .title-text {
        font-size: 2.45rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: 0.01em;
        line-height: 1.08;
        margin-top: 0.1rem;
        margin-bottom: 0.35rem;
    }

    .subtitle-text {
        color: #94a3b8;
        font-size: 1.02rem;
        margin-top: 0;
        margin-bottom: 0.85rem;
    }

    .trust-box {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 0.8rem;
        color: #cbd5e1;
        font-size: 0.94rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
def safe_read_csv(path: Path, index_col=0, parse_dates=True):
    if path.exists():
        return pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)
    return None


def compute_metrics(return_series: pd.Series) -> dict:
    if return_series is None:
        return {
            "CAGR": np.nan,
            "Annual Vol": np.nan,
            "Sharpe": np.nan,
            "Max Drawdown": np.nan,
            "Calmar": np.nan,
        }

    rs = return_series.dropna()
    if len(rs) == 0:
        return {
            "CAGR": np.nan,
            "Annual Vol": np.nan,
            "Sharpe": np.nan,
            "Max Drawdown": np.nan,
            "Calmar": np.nan,
        }

    cumulative = (1 + rs).cumprod()
    months = len(rs)

    cagr = cumulative.iloc[-1] ** (12 / months) - 1
    vol = rs.std() * np.sqrt(12)
    sharpe = cagr / vol if vol and not np.isnan(vol) else np.nan

    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd and not np.isnan(max_dd) else np.nan

    return {
        "CAGR": cagr,
        "Annual Vol": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
    }


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


def normalize_returns(obj):
    if isinstance(obj, pd.Series):
        s = obj.dropna().copy()
        out = (1 + s).cumprod()
        out = out / out.iloc[0]
        return out

    df = obj.dropna(how="all").copy()
    out = (1 + df).cumprod()
    out = out / out.iloc[0]
    return out


def slugify_column_name(column_name: str) -> str:
    return column_name.lower()


def to_monthly_return_table(return_series: pd.Series) -> pd.DataFrame:
    s = return_series.dropna().copy()
    df = s.to_frame("Return")
    df["Year"] = df.index.year
    df["Month"] = df.index.month_name().str[:3]
    pivot = df.pivot(index="Year", columns="Month", values="Return")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=month_order)
    return pivot.sort_index()


def run_bootstrap_monte_carlo(return_series, horizon_months=60, n_sims=25000, seed=42):
    rs = return_series.dropna().values
    rng = np.random.default_rng(seed)

    sim_paths = np.zeros((horizon_months + 1, n_sims))
    sim_paths[0, :] = 1.0
    max_drawdowns = np.zeros(n_sims)

    for j in range(n_sims):
        sampled = rng.choice(rs, size=horizon_months, replace=True)
        path = np.concatenate([[1.0], np.cumprod(1 + sampled)])
        sim_paths[:, j] = path

        running_max = np.maximum.accumulate(path)
        drawdowns = path / running_max - 1
        max_drawdowns[j] = drawdowns.min()

    plot_df = pd.DataFrame({
        "p10": np.percentile(sim_paths, 10, axis=1),
        "p50": np.percentile(sim_paths, 50, axis=1),
        "p90": np.percentile(sim_paths, 90, axis=1),
    }, index=range(horizon_months + 1))

    terminal_values = sim_paths[-1, :]

    summary = {
        "median_terminal_value": np.median(terminal_values),
        "p10_terminal_value": np.percentile(terminal_values, 10),
        "p90_terminal_value": np.percentile(terminal_values, 90),
        "prob_finish_above_1_0x": np.mean(terminal_values > 1.0),
        "prob_finish_above_1_5x": np.mean(terminal_values > 1.5),
        "prob_finish_above_2_0x": np.mean(terminal_values > 2.0),
        "median_max_drawdown": np.median(max_drawdowns),
        "prob_maxdd_worse_than_20pct": np.mean(max_drawdowns < -0.20),
        "prob_maxdd_worse_than_30pct": np.mean(max_drawdowns < -0.30),
        "horizon_months": horizon_months,
        "num_simulations": n_sims,
    }

    terminal_df = pd.DataFrame({"terminal_value": terminal_values})
    drawdown_df = pd.DataFrame({"max_drawdown": max_drawdowns})

    return plot_df, summary, terminal_df, drawdown_df


# ---------- Load centralized app data ----------
app_strategy_returns = safe_read_csv(APP_DATA_DIR / "app_strategy_returns.csv")
app_strategy_metrics = safe_read_csv(APP_DATA_DIR / "app_strategy_metrics.csv")
app_strategy_registry = safe_read_csv(APP_DATA_DIR / "app_strategy_registry.csv", index_col=None, parse_dates=False)
app_config = safe_read_csv(APP_DATA_DIR / "app_config.csv", index_col=None, parse_dates=False)

if app_strategy_returns is None or app_strategy_registry is None:
    st.error(
        "App data files are missing. Run 12_app_data_hub.ipynb first so the dashboard can read from data/processed/app_data/."
    )
    st.stop()

# ---------- Registry prep ----------
registry_df = app_strategy_registry.copy()
registry_df["is_default"] = registry_df["is_default"].astype(bool)

strategy_rows = registry_df[registry_df["category"] != "benchmark"].copy()
benchmark_rows = registry_df[registry_df["category"] == "benchmark"].copy()

strategy_display_to_column = dict(zip(strategy_rows["display_name"], strategy_rows["column_name"]))
benchmark_display_to_column = dict(zip(benchmark_rows["display_name"], benchmark_rows["column_name"]))
column_to_display = dict(zip(registry_df["column_name"], registry_df["display_name"]))
column_to_description = dict(zip(registry_df["column_name"], registry_df["description"]))

available_strategy_display_names = [
    d for d, c in strategy_display_to_column.items() if c in app_strategy_returns.columns
]

if len(available_strategy_display_names) == 0:
    st.error("No registered app strategies were found in app_strategy_returns.csv")
    st.stop()

if app_config is not None and "recommended_display_name" in app_config.columns:
    recommended_display_name = app_config.loc[0, "recommended_display_name"]
else:
    default_rows = strategy_rows[strategy_rows["is_default"]]
    recommended_display_name = (
        default_rows["display_name"].iloc[0]
        if len(default_rows) > 0
        else available_strategy_display_names[0]
    )

# ---------- Sidebar ----------
st.sidebar.markdown("## Strategy Controls")

default_idx = (
    available_strategy_display_names.index(recommended_display_name)
    if recommended_display_name in available_strategy_display_names
    else 0
)

selected_strategy_display = st.sidebar.selectbox(
    "Strategy Engine",
    options=available_strategy_display_names,
    index=default_idx,
)
selected_col = strategy_display_to_column[selected_strategy_display]

available_benchmark_display_names = [
    d for d, c in benchmark_display_to_column.items() if c in app_strategy_returns.columns
]

default_benchmarks = [
    name for name in ["SPY Buy & Hold", "60/40 Portfolio", "DBMF", "KMLM", "RPAR"]
    if name in available_benchmark_display_names
]

selected_benchmark_display_names = st.sidebar.multiselect(
    "Benchmarks",
    options=available_benchmark_display_names,
    default=default_benchmarks,
)
selected_benchmark_cols = [benchmark_display_to_column[d] for d in selected_benchmark_display_names]

show_debug = st.sidebar.checkbox("Show debug info", value=False)

# ---------- Header ----------
st.markdown("<div class='title-text'>Quant Portfolio Dashboard</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle-text'>Centralized app view for the final strategy engines, benchmarks, weights, diagnostics, and robustness checks.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='trust-box'>
    <b>Research setup.</b> Final strategies were developed using a train/validation/test workflow, then benchmarked on the common overlapping benchmark window. 
    The chart below is aligned to the first date where both the selected strategy and chosen benchmarks all have data, so the comparison starts on the same date for every displayed series.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Selected series ----------
strategy_series = app_strategy_returns[selected_col] if selected_col in app_strategy_returns.columns else None
strategy_metrics = compute_metrics(strategy_series)

start_date = strategy_series.dropna().index.min().date() if strategy_series is not None and len(strategy_series.dropna()) > 0 else "—"
end_date = strategy_series.dropna().index.max().date() if strategy_series is not None and len(strategy_series.dropna()) > 0 else "—"
strategy_description = column_to_description.get(selected_col, "")

compare_cols = [selected_col] + [c for c in selected_benchmark_cols if c in app_strategy_returns.columns]
raw_compare = app_strategy_returns[compare_cols].copy()

strategy_start = app_strategy_returns[selected_col].dropna().index.min()
raw_compare = raw_compare.loc[strategy_start:]
raw_compare = raw_compare.dropna()

comparison_start = raw_compare.index.min()
comparison_end = raw_compare.index.max()

comparison_df = normalize_returns(raw_compare)
display_df = comparison_df.rename(columns=column_to_display)
comparison_growth_value = comparison_df[selected_col].iloc[-1]

# ---------- Top metric cards ----------
metric_cols = st.columns(5, gap="medium")
with metric_cols[0]:
    draw_metric_card("Selected Strategy", selected_strategy_display, f"Backtest window: {start_date} → {end_date}")
with metric_cols[1]:
    draw_metric_card("Growth of $1", fmt_num(comparison_growth_value), f"Growth since {comparison_start.date()}")
with metric_cols[2]:
    draw_metric_card("CAGR", fmt_pct(strategy_metrics["CAGR"]), "Annualized return")
with metric_cols[3]:
    draw_metric_card("Sharpe", fmt_num(strategy_metrics["Sharpe"]), "Risk-adjusted return")
with metric_cols[4]:
    draw_metric_card("Max Drawdown", fmt_pct(strategy_metrics["Max Drawdown"]), "Worst peak-to-trough decline")

st.caption(strategy_description)

# ---------- Performance row ----------
row1_left, row1_right = st.columns([1.65, 1], gap="large")

with row1_left:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Performance Comparison")
    st.line_chart(display_df, height=390)
    st.caption(
        f"Normalized cumulative performance from the common comparison window: {comparison_start.date()} to {comparison_end.date()}."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with row1_right:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Strategy Metrics")

    summary_cols = compare_cols
    if app_strategy_metrics is not None and set(summary_cols).issubset(app_strategy_metrics.index):
        metrics_table = app_strategy_metrics.loc[summary_cols].copy()
    else:
        metrics_table = pd.DataFrame({col: compute_metrics(app_strategy_returns[col]) for col in summary_cols}).T

    metrics_table.index = [column_to_display.get(c, c) for c in metrics_table.index]

    st.dataframe(
        metrics_table.style.format({
            "CAGR": "{:.2%}",
            "Annual Vol": "{:.2%}",
            "Sharpe": "{:.2f}",
            "Max Drawdown": "{:.2%}",
            "Calmar": "{:.2f}",
        }),
        use_container_width=True,
        height=390,
    )
    st.caption("These metrics use each series’ full available return history unless otherwise stated.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Weights row ----------
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("Current / Latest Weights")

weight_file = APP_DATA_DIR / f"app_weights_{slugify_column_name(selected_col)}.csv"
selected_weight_df = safe_read_csv(weight_file)

if selected_weight_df is not None:
    latest_weights = selected_weight_df.iloc[-1].dropna()
    latest_weights = latest_weights[latest_weights > 0].sort_values(ascending=False)

    if len(latest_weights) > 0:
        weights_left, weights_right = st.columns([1.1, 1], gap="large")

        with weights_left:
            latest_df = latest_weights.rename("Weight").to_frame()
            st.bar_chart(latest_df, height=320)

        with weights_right:
            asset_table = latest_weights.rename("Weight").reset_index()
            asset_table.columns = ["Asset", "Weight"]
            st.dataframe(
                asset_table.style.format({"Weight": "{:.2%}"}),
                use_container_width=True,
                height=320,
            )

        st.caption(f"Latest saved portfolio weights as of {selected_weight_df.index.max().date()}.")
    else:
        st.info("No positive weights were found in the latest saved row for this strategy.")
else:
    st.warning(f"Missing app weight file: {weight_file.name}")

st.markdown("</div>", unsafe_allow_html=True)

# ---------- Drawdown row ----------
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("Drawdown")
dd = comparison_df.div(comparison_df.cummax()) - 1
st.line_chart(dd.rename(columns=column_to_display), height=320)
st.caption("Peak-to-trough drawdown using the same aligned comparison window.")
st.markdown("</div>", unsafe_allow_html=True)

# ---------- Volatility row ----------
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("Rolling Volatility")
roll_vol = raw_compare.rolling(12).std() * np.sqrt(12)
st.line_chart(roll_vol.rename(columns=column_to_display), height=320)
st.caption("12-month rolling annualized volatility using the same aligned comparison window.")
st.markdown("</div>", unsafe_allow_html=True)

# ---------- Monte Carlo row ----------
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("Monte Carlo Robustness")

horizon_months = 60
n_sims = 25000

mc_plot_df, mc_summary, mc_terminal_df, mc_drawdown_df = run_bootstrap_monte_carlo(
    strategy_series,
    horizon_months=horizon_months,
    n_sims=n_sims,
    seed=42,
)

st.caption(
    f"Bootstrap Monte Carlo for the selected strategy | Horizon: {horizon_months} months ({horizon_months/12:.0f} years) | Simulations: {n_sims:,}"
)

mc_top_left, mc_top_right = st.columns([1.2, 1], gap="large")

with mc_top_left:
    st.line_chart(mc_plot_df[["p10", "p50", "p90"]], height=240)

with mc_top_right:
    mc_metrics = pd.DataFrame({
        "Metric": [
            "Median terminal value",
            "10th percentile terminal",
            "90th percentile terminal",
            "Prob finish above 1.0x",
            "Prob finish above 1.5x",
            "Prob finish above 2.0x",
            "Median max drawdown",
            "Prob max DD worse than 20%",
            "Prob max DD worse than 30%",
        ],
        "Value": [
            f"{mc_summary['median_terminal_value']:.2f}",
            f"{mc_summary['p10_terminal_value']:.2f}",
            f"{mc_summary['p90_terminal_value']:.2f}",
            f"{mc_summary['prob_finish_above_1_0x']:.2%}",
            f"{mc_summary['prob_finish_above_1_5x']:.2%}",
            f"{mc_summary['prob_finish_above_2_0x']:.2%}",
            f"{mc_summary['median_max_drawdown']:.2%}",
            f"{mc_summary['prob_maxdd_worse_than_20pct']:.2%}",
            f"{mc_summary['prob_maxdd_worse_than_30pct']:.2%}",
        ],
    })
    st.dataframe(mc_metrics, use_container_width=True, height=240)

mc_bottom_left, mc_bottom_right = st.columns(2, gap="large")

with mc_bottom_left:
    st.markdown("**Terminal Value Distribution**")
    st.bar_chart(
        mc_terminal_df["terminal_value"].value_counts(bins=30, sort=False),
        height=220
    )

with mc_bottom_right:
    st.markdown("**Max Drawdown Distribution**")
    st.bar_chart(
        mc_drawdown_df["max_drawdown"].value_counts(bins=30, sort=False),
        height=220
    )

st.caption(
    "This Monte Carlo resamples the selected strategy’s historical monthly returns. It is a robustness tool, not a forecast guarantee."
)
st.markdown("</div>", unsafe_allow_html=True)

# ---------- Monthly history / latest holdings row ----------
row4_left, row4_right = st.columns([1, 1], gap="large")

with row4_left:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Monthly Return History")

    monthly_table = to_monthly_return_table(strategy_series)
    st.dataframe(
        monthly_table.style.format("{:.2%}", na_rep=""),
        use_container_width=True,
        height=330,
    )
    st.caption("Calendar table of the selected strategy’s monthly returns.")
    st.markdown("</div>", unsafe_allow_html=True)

with row4_right:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Latest Invested Assets")

    if selected_weight_df is not None:
        latest_weights = selected_weight_df.iloc[-1].dropna()
        latest_weights = latest_weights[latest_weights > 0].sort_values(ascending=False)
        if len(latest_weights) > 0:
            asset_table = latest_weights.rename("Weight").reset_index()
            asset_table.columns = ["Asset", "Weight"]
            st.dataframe(
                asset_table.style.format({"Weight": "{:.2%}"}),
                use_container_width=True,
                height=330,
            )
        else:
            st.info("No positive invested assets found in the latest row.")
    else:
        st.info("No weight history available for this strategy yet.")

    st.caption("This shows what the strategy most recently held based on the saved app weights file.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Diagnostics ----------
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("Research / Diagnostics")

diag_tab1, diag_tab2, diag_tab3, diag_tab4 = st.tabs(
    ["Registry", "Return Correlations", "Benchmarks", "Recommended Engine"]
)

with diag_tab1:
    st.dataframe(registry_df, use_container_width=True)

with diag_tab2:
    corr = raw_compare.corr()
    corr.index = [column_to_display.get(c, c) for c in corr.index]
    corr.columns = [column_to_display.get(c, c) for c in corr.columns]
    st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)

with diag_tab3:
    benchmark_metric_cols = [c for c in benchmark_display_to_column.values() if c in app_strategy_returns.columns]
    if len(benchmark_metric_cols) > 0:
        benchmark_metrics = (
            app_strategy_metrics.loc[benchmark_metric_cols].copy()
            if app_strategy_metrics is not None and set(benchmark_metric_cols).issubset(app_strategy_metrics.index)
            else pd.DataFrame({col: compute_metrics(app_strategy_returns[col]) for col in benchmark_metric_cols}).T
        )
        benchmark_metrics.index = [column_to_display.get(c, c) for c in benchmark_metrics.index]
        st.dataframe(
            benchmark_metrics.style.format({
                "CAGR": "{:.2%}",
                "Annual Vol": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Max Drawdown": "{:.2%}",
                "Calmar": "{:.2f}",
            }),
            use_container_width=True,
        )
    else:
        st.info("No benchmark series were found in app_strategy_returns.csv")

with diag_tab4:
    if app_config is not None and len(app_config) > 0:
        st.write(f"**Recommended Default:** {app_config.loc[0, 'recommended_display_name']}")
        st.write(app_config.loc[0, 'recommended_reason'])
    else:
        st.write(f"**Recommended Default:** {recommended_display_name}")
        st.write("Best overall balance of CAGR, Sharpe, and drawdown among the final app engines.")

st.markdown("</div>", unsafe_allow_html=True)

# ---------- Debug ----------
if show_debug:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Debug Info")
    st.write("APP_DATA_DIR:", str(APP_DATA_DIR))
    st.write("Selected column:", selected_col)
    st.write("Comparison start:", comparison_start)
    st.write("Available app columns:", app_strategy_returns.columns.tolist())
    st.write("Expected weight file:", f"app_weights_{slugify_column_name(selected_col)}.csv")
    st.write("Monte Carlo horizon (months):", 60)
    st.write("Monte Carlo simulations:", 25000)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Footer ----------
st.caption(
    "Next steps: refine chart aesthetics, polish explanatory language, and keep the app pointed only at centralized app_data outputs."
)