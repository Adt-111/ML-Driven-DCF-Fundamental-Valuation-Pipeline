"""
ml_dcf_valuation.src
=====================
Modular components for the ML-Driven DCF & Fundamental Valuation Pipeline.

Modules
-------
data_fetcher   : Pulls raw financial statements & market data (yfinance).
ml_forecaster  : XGBoost/LightGBM driver forecasts (Revenue Growth, EBIT %, CapEx %).
wacc_engine    : CAPM, Cost of Debt, and WACC computation.
dcf_calculator : Two-stage unlevered FCFF DCF valuation engine.
monte_carlo    : Sensitivity analysis & Monte Carlo simulation of intrinsic value.
"""

__version__ = "1.0.0"
