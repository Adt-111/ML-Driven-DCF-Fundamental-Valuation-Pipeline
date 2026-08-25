"""
Streamlit dashboard for the ML-Driven DCF & Fundamental Valuation Pipeline.

Run with:
    streamlit run app.py

Features
--------
- Ticker input (e.g. AAPL, NVDA, MSFT) with a "Run Valuation" trigger.
- Historical fundamentals vs. ML-forecast drivers (Revenue Growth, EBIT
  Margin, CapEx % of Revenue).
- WACC breakdown (Beta, Cost of Equity, Cost of Debt, WACC).
- DCF intrinsic value vs. current market price.
- Monte Carlo simulation histogram (Plotly) of implied share price.
- WACC x Terminal Growth sensitivity table.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_fetcher import DataFetcher, fetch_market_series  # noqa: E402
from ml_forecaster import MLForecaster  # noqa: E402
from wacc_engine import WACCEngine  # noqa: E402
from dcf_calculator import DCFCalculator  # noqa: E402
from monte_carlo import MonteCarloSimulator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("app")

st.set_page_config(page_title="ML-Driven DCF Valuation", layout="wide", page_icon="📈")


# cached pipeline stages (avoid re-fetching / re-training on every widget tick)
@st.cache_data(ttl=3600, show_spinner=False)
def load_financials(ticker: str):
    fetcher = DataFetcher(ticker)
    return fetcher.fetch_all()


@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(ticker: str):
    stock_px = fetch_market_series(ticker)
    mkt_px = fetch_market_series("^GSPC")
    tnx = fetch_market_series("^TNX")
    return stock_px, mkt_px, tnx


def run_pipeline(ticker: str, erp: float, terminal_growth: float, n_sims: int):
    """Executes the full fetch -> forecast -> WACC -> DCF -> Monte Carlo pipeline."""
    financials = load_financials(ticker)
    if financials.income_statement.empty:
        return None

    revenue_row = MLForecaster._row(
        financials.income_statement, ["Total Revenue", "Operating Revenue"]
    )

    forecaster = MLForecaster()
    table = forecaster.build_training_table(financials.income_statement)
    table = forecaster.attach_capex_ratio(table, financials.cash_flow, revenue_row)
    forecast = forecaster.fit_and_forecast(table)

    stock_px, mkt_px, tnx = load_market_data(ticker)
    interest_expense = 0.0
    if "Interest Expense" in financials.income_statement.index:
        interest_expense = abs(float(financials.income_statement.loc["Interest Expense"].iloc[-1]))

    wacc_engine = WACCEngine(equity_risk_premium=erp)
    wacc_result = wacc_engine.run(
        stock_prices=stock_px,
        market_prices=mkt_px,
        treasury_yield_series=tnx,
        market_cap=financials.info.get("marketCap", np.nan),
        total_debt=financials.total_debt,
        interest_expense=interest_expense,
        tax_rate=financials.effective_tax_rate,
    )

    base_revenue = float(revenue_row.iloc[-1])
    dcf = DCFCalculator(terminal_growth_rate=terminal_growth)
    dcf_result = dcf.run(
        base_revenue=base_revenue,
        revenue_growth=forecast.revenue_growth,
        ebit_margin=forecast.ebit_margin,
        capex_pct_revenue=forecast.capex_pct_revenue,
        tax_rate=financials.effective_tax_rate,
        wacc=wacc_result.wacc,
        total_debt=financials.total_debt,
        cash_and_equivalents=financials.cash_and_equivalents,
        diluted_shares=financials.shares_outstanding,
        current_price=financials.current_price,
        forecast_years=forecast.years,
    )

    mc = MonteCarloSimulator(n_simulations=n_sims)
    mc_result = mc.run(
        base_revenue=base_revenue,
        revenue_growth=forecast.revenue_growth,
        ebit_margin=forecast.ebit_margin,
        capex_pct_revenue=forecast.capex_pct_revenue,
        tax_rate=financials.effective_tax_rate,
        wacc=wacc_result.wacc,
        total_debt=financials.total_debt,
        cash_and_equivalents=financials.cash_and_equivalents,
        diluted_shares=financials.shares_outstanding,
        current_price=financials.current_price,
        forecast_years=forecast.years,
        terminal_growth_rate=terminal_growth,
    )

    sensitivity = mc.sensitivity_grid(
        base_revenue=base_revenue,
        revenue_growth=forecast.revenue_growth,
        ebit_margin=forecast.ebit_margin,
        capex_pct_revenue=forecast.capex_pct_revenue,
        tax_rate=financials.effective_tax_rate,
        total_debt=financials.total_debt,
        cash_and_equivalents=financials.cash_and_equivalents,
        diluted_shares=financials.shares_outstanding,
        current_price=financials.current_price,
        forecast_years=forecast.years,
        wacc_center=wacc_result.wacc,
        terminal_growth_center=terminal_growth,
    )

    return {
        "financials": financials,
        "forecast": forecast,
        "training_table": table,
        "wacc_result": wacc_result,
        "dcf_result": dcf_result,
        "mc_result": mc_result,
        "sensitivity": sensitivity,
    }


# sidebar controls
st.sidebar.title("📈 ML-DCF Valuation")
st.sidebar.caption("ML-Driven DCF & Fundamental Valuation Pipeline")

ticker_input = st.sidebar.text_input("Ticker", value="AAPL", help="e.g. AAPL, NVDA, MSFT").upper().strip()
erp_input = st.sidebar.slider("Equity Risk Premium (%)", 3.0, 7.0, 5.0, 0.1) / 100
terminal_growth_input = st.sidebar.slider("Terminal Growth Rate (%)", 0.5, 4.0, 2.5, 0.1) / 100
n_sims_input = st.sidebar.select_slider("Monte Carlo Simulations", options=[500, 1000, 2500, 5000, 10000], value=2500)
run_button = st.sidebar.button("Run Valuation", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ For educational/research purposes only. Not investment advice. "
    "Data sourced via yfinance and may be delayed or incomplete."
)

st.title("ML-Driven DCF & Fundamental Valuation")

if "results" not in st.session_state:
    st.session_state.results = None
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = None

if run_button or (st.session_state.results is None and ticker_input):
    with st.spinner(f"Running valuation pipeline for {ticker_input}..."):
        try:
            st.session_state.results = run_pipeline(
                ticker_input, erp_input, terminal_growth_input, n_sims_input
            )
            st.session_state.last_ticker = ticker_input
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline failed")
            st.error(f"Valuation pipeline failed for {ticker_input}: {exc}")
            st.session_state.results = None

results = st.session_state.results

if results is None:
    st.info("Enter a ticker in the sidebar and click **Run Valuation** to begin.")
    st.stop()

financials = results["financials"]
forecast = results["forecast"]
wacc_result = results["wacc_result"]
dcf_result = results["dcf_result"]
mc_result = results["mc_result"]
sensitivity = results["sensitivity"]

st.subheader(f"{financials.ticker} - {financials.info.get('shortName', '')}")

# top-line kpi row
kpi_cols = st.columns(5)
kpi_cols[0].metric("Current Price", f"${financials.current_price:,.2f}")
kpi_cols[1].metric(
    "DCF Implied Price",
    f"${dcf_result.implied_share_price:,.2f}",
    f"{dcf_result.upside_downside_pct:+.1f}%",
)
kpi_cols[2].metric("WACC", f"{wacc_result.wacc*100:.2f}%")
kpi_cols[3].metric("Beta (36M Rolling)", f"{wacc_result.beta:.2f}")
kpi_cols[4].metric("Monte Carlo Median", f"${mc_result.median_price:,.2f}")

st.markdown("---")

# historical vs. ml forecast drivers
st.subheader("Historical Fundamentals vs. ML-Forecast Drivers")

hist_table = results["training_table"][
    ["year", "revenue_growth", "ebit_margin", "capex_pct_revenue"]
].copy()
hist_table["type"] = "Historical"
hist_table = hist_table.rename(
    columns={"year": "Year", "revenue_growth": "Revenue Growth",
             "ebit_margin": "EBIT Margin", "capex_pct_revenue": "CapEx % Rev"}
)

fc_table = forecast.to_frame().rename(
    columns={"Revenue Growth (%)": "Revenue Growth", "EBIT Margin (%)": "EBIT Margin",
             "CapEx / Revenue (%)": "CapEx % Rev"}
)
fc_table[["Revenue Growth", "EBIT Margin", "CapEx % Rev"]] /= 100
fc_table["type"] = "ML Forecast"

combined = pd.concat([hist_table, fc_table], ignore_index=True)

driver_cols = st.columns(3)
driver_labels = ["Revenue Growth", "EBIT Margin", "CapEx % Rev"]
for col, label in zip(driver_cols, driver_labels):
    fig = go.Figure()
    for series_type, color in [("Historical", "#636EFA"), ("ML Forecast", "#EF553B")]:
        sub = combined[combined["type"] == series_type]
        fig.add_trace(
            go.Scatter(
                x=sub["Year"], y=sub[label] * 100, mode="lines+markers",
                name=series_type, line=dict(color=color),
            )
        )
    fig.update_layout(
        title=label, height=320, margin=dict(l=10, r=10, t=40, b=10),
        yaxis_title="%", showlegend=(label == "Revenue Growth"),
    )
    col.plotly_chart(fig, use_container_width=True)

st.caption(f"Forecast model: **{forecast.model_used}** | Forecast horizon: {len(forecast.years)} years")

st.markdown("---")

# wacc breakdown
st.subheader("WACC Build-Up (CAPM)")
wacc_cols = st.columns(6)
wacc_cols[0].metric("Risk-Free Rate", f"{wacc_result.risk_free_rate*100:.2f}%")
wacc_cols[1].metric("Beta", f"{wacc_result.beta:.2f}")
wacc_cols[2].metric("Equity Risk Premium", f"{wacc_result.equity_risk_premium*100:.2f}%")
wacc_cols[3].metric("Cost of Equity", f"{wacc_result.cost_of_equity*100:.2f}%")
wacc_cols[4].metric("Cost of Debt (pre-tax)", f"{wacc_result.cost_of_debt*100:.2f}%")
wacc_cols[5].metric("WACC", f"{wacc_result.wacc*100:.2f}%")

st.caption(
    f"Capital structure weights - Equity: {wacc_result.weight_equity*100:.1f}% | "
    f"Debt: {wacc_result.weight_debt*100:.1f}% | Tax Rate: {wacc_result.tax_rate*100:.1f}%"
)

st.markdown("---")

# dcf output
st.subheader("2-Stage FCFF DCF Valuation")

dcf_c1, dcf_c2 = st.columns([2, 1])

with dcf_c1:
    summary_df = dcf_result.summary_frame()
    fig_fcff = go.Figure()
    fig_fcff.add_trace(go.Bar(x=summary_df["Year"], y=summary_df["Projected FCFF"] / 1e6, name="Projected FCFF ($M)"))
    fig_fcff.add_trace(go.Bar(x=summary_df["Year"], y=summary_df["PV of FCFF"] / 1e6, name="PV of FCFF ($M)"))
    fig_fcff.update_layout(
        barmode="group", height=380, title="Projected & Discounted FCFF",
        yaxis_title="$ Millions", margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_fcff, use_container_width=True)

with dcf_c2:
    st.markdown("**Valuation Bridge**")
    st.write(f"Sum PV of FCFF: ${np.sum(dcf_result.pv_fcff)/1e6:,.1f}M")
    st.write(f"PV of Terminal Value: ${dcf_result.pv_terminal_value/1e6:,.1f}M")
    st.write(f"**Enterprise Value: ${dcf_result.enterprise_value/1e6:,.1f}M**")
    st.write(f"Less: Net Debt: ${dcf_result.net_debt/1e6:,.1f}M")
    st.write(f"**Equity Value: ${dcf_result.equity_value/1e6:,.1f}M**")
    st.write(f"Diluted Shares: {dcf_result.diluted_shares/1e6:,.1f}M")
    st.write(f"### Implied Price: ${dcf_result.implied_share_price:,.2f}")
    delta_color = "normal" if dcf_result.upside_downside_pct >= 0 else "inverse"
    st.metric("vs. Current Price", f"${dcf_result.current_price:,.2f}", f"{dcf_result.upside_downside_pct:+.1f}%")

with st.expander("View detailed year-by-year FCFF schedule"):
    st.dataframe(summary_df.style.format({
        "Projected Revenue": "${:,.0f}",
        "Projected FCFF": "${:,.0f}",
        "Discount Factor": "{:.4f}",
        "PV of FCFF": "${:,.0f}",
    }), use_container_width=True)

st.markdown("---")

# monte carlo simulation
st.subheader("Monte Carlo Valuation Simulation")

mc_c1, mc_c2 = st.columns([2, 1])

with mc_c1:
    if len(mc_result.simulated_prices) > 0:
        fig_mc = go.Figure()
        fig_mc.add_trace(
            go.Histogram(
                x=mc_result.simulated_prices, nbinsx=60, marker_color="#00CC96",
                name="Implied Price Distribution",
            )
        )
        fig_mc.add_vline(
            x=dcf_result.current_price, line_dash="dash", line_color="red",
            annotation_text="Current Price",
        )
        fig_mc.add_vline(
            x=mc_result.median_price, line_dash="dash", line_color="blue",
            annotation_text="Median Simulated",
        )
        fig_mc.update_layout(
            height=380, title=f"Monte Carlo Distribution of Implied Share Price ({len(mc_result.simulated_prices):,} sims)",
            xaxis_title="Implied Share Price ($)", yaxis_title="Frequency",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_mc, use_container_width=True)
    else:
        st.warning("Monte Carlo simulation produced no valid results for this ticker.")

with mc_c2:
    st.markdown("**Simulation Statistics**")
    stats = mc_result.summary()
    for label, value in stats.items():
        if label == "P(Upside)":
            st.write(f"{label}: {value*100:.1f}%")
        else:
            st.write(f"{label}: ${value:,.2f}")

st.markdown("---")

# sensitivity table
st.subheader("Sensitivity: WACC vs. Terminal Growth Rate")
st.caption("Rows = WACC, Columns = Terminal Growth Rate. Cell values = implied share price.")
st.dataframe(
    sensitivity.style.format("${:,.2f}", na_rep="-").background_gradient(cmap="RdYlGn", axis=None),
    use_container_width=True,
)

st.markdown("---")
st.caption(
    "Methodology: FCFF = EBIT×(1−t) + D&A − CapEx − ΔNWC | "
    "WACC via CAPM with 36-month rolling beta vs. ^GSPC, R_f from ^TNX | "
    "Terminal Value via Gordon Growth Model | Drivers forecast via gradient-boosted trees "
    "(XGBoost/LightGBM) with historical-mean shrinkage."
)
