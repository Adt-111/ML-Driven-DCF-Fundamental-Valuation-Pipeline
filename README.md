# ML-Driven DCF & Fundamental Valuation Pipeline

A modular, production-style Python pipeline that combines fundamental
financial-statement analysis, machine-learning driver forecasting, CAPM-based
WACC estimation, and a two-stage FCFF DCF model — wrapped in an interactive
Streamlit dashboard with Monte Carlo valuation simulation.

> ⚠️ **Disclaimer:** This project is for educational and research purposes
> only. It does not constitute investment advice. Market data via `yfinance`
> may be delayed, incomplete, or occasionally missing line items depending on
> the ticker.

## Project Structure

```
ml_dcf_valuation/
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py       # Pulls financial statements via yfinance
│   ├── ml_forecaster.py      # XGBoost/LightGBM driver forecasts
│   ├── wacc_engine.py        # CAPM, Cost of Debt, WACC
│   ├── dcf_calculator.py     # 2-Stage Unlevered FCFF DCF
│   └── monte_carlo.py        # Sensitivity & Monte Carlo simulation
└── app.py                    # Streamlit dashboard
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Dashboard

```bash
streamlit run app.py
```

Then enter a ticker (e.g. `AAPL`, `NVDA`, `MSFT`) in the sidebar and click
**Run Valuation**.

## Running Individual Modules (CLI smoke tests)

Each module has a `if __name__ == "__main__":` block for standalone testing:

```bash
cd src
python data_fetcher.py       # prints historical FCFF table
python ml_forecaster.py      # prints 5-year driver forecast
python wacc_engine.py        # prints WACC breakdown
python dcf_calculator.py     # prints DCF schedule + implied price
python monte_carlo.py        # prints Monte Carlo summary stats
```

## Methodology

### 1. Historical FCFF
```
FCFF = EBIT × (1 − Tax Rate) + D&A − CapEx − ΔNWC
```
Pulled from `yfinance` Income Statement / Balance Sheet / Cash Flow
Statement, trimmed to the most recent 5 annual periods.

### 2. ML Driver Forecasting
Gradient-boosted regressors (XGBoost, with LightGBM/Ridge fallback) trained
on lag-based features (prior-year ratio, trailing mean) to recursively
forecast 5 years of:
- Revenue Growth (%)
- EBIT Margin (%)
- CapEx / Revenue (%)

Predictions are shrunk toward the trailing historical mean to avoid
overfitting on short annual histories (Bayesian-style blending).

### 3. WACC
```
r_e (CAPM) = R_f + β × ERP
r_d        = Interest Expense / Total Debt  (or R_f + credit spread fallback)
WACC       = (E/V)×r_e + (D/V)×r_d×(1 − t)
```
- **β**: 36-month rolling OLS regression of monthly stock returns vs. S&P 500
  (`^GSPC`).
- **R_f**: Most recent 10-Year U.S. Treasury yield (`^TNX`).

### 4. Two-Stage DCF
- **Stage 1**: Explicit 5-year FCFF projection discounted at WACC.
- **Stage 2**: Terminal Value via Gordon Growth Model:
  `TV = FCFF_n × (1+g) / (WACC − g)`
- **Enterprise Value** = ΣPV(FCFF) + PV(TV)
- **Equity Value** = Enterprise Value − Net Debt (Total Debt − Cash)
- **Implied Share Price** = Equity Value / Diluted Shares Outstanding

### 5. Monte Carlo & Sensitivity
Randomly perturbs revenue growth, EBIT margin, WACC, and terminal growth
across thousands of iterations to build a distribution of implied share
prices, plus a WACC × terminal-growth 2D sensitivity grid.

## Extending the Pipeline

- **Alternative data source**: `data_fetcher.py` includes an `FMPDataFetcher`
  stub implementing the same interface as `DataFetcher` for Financial
  Modeling Prep API integration.
- **Macro indicators**: `ml_forecaster.fetch_macro_indicators()` is a
  placeholder for FRED API series (GDP growth, CPI); wire in `fredapi` for
  live macro features.
- **Cross-sectional training**: For materially better ML forecasts, train
  driver models across a peer universe (sector/industry) rather than a
  single ticker's short annual history.

## Known Limitations

- `yfinance` typically exposes only ~4-5 years of annual statements, which
  constrains per-company ML training data (mitigated via shrinkage).
- Quarterly-frequency modeling is not implemented; all statements are annual.
- Equity Risk Premium and default credit spread are user-configurable
  assumptions, not market-implied.
