"""
Two-Stage Unlevered Free Cash Flow to the Firm (FCFF) DCF valuation model.

Stage 1 (explicit forecast, years 1-5):
    FCFF_t = Revenue_t * EBIT_Margin_t * (1 - tax) + D&A_t - CapEx_t - Delta_NWC_t
    (simplified here to NOPAT - reinvestment, since D&A/CapEx/NWC are
    folded into a single "reinvestment rate" proxy when only margin-level
    ML forecasts are available - see `project_fcff`)

Stage 2 (terminal value, Gordon Growth Model):
    TV_n = FCFF_n * (1 + g) / (WACC - g)

Enterprise Value = sum(PV(FCFF_t)) + PV(TV_n)
Equity Value     = Enterprise Value - Net Debt  (Net Debt = Total Debt - Cash)
Implied Share Price = Equity Value / Diluted Shares Outstanding
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DCFResult:
    """Full breakdown of the DCF valuation for transparency and reporting."""

    projection_years: list[int]
    projected_revenue: np.ndarray
    projected_fcff: np.ndarray
    discount_factors: np.ndarray
    pv_fcff: np.ndarray
    terminal_value: float
    pv_terminal_value: float
    enterprise_value: float
    net_debt: float
    equity_value: float
    diluted_shares: float
    implied_share_price: float
    current_price: float
    upside_downside_pct: float
    wacc: float
    terminal_growth_rate: float

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Year": self.projection_years,
                "Projected Revenue": self.projected_revenue,
                "Projected FCFF": self.projected_fcff,
                "Discount Factor": self.discount_factors,
                "PV of FCFF": self.pv_fcff,
            }
        )


class DCFCalculator:
    """
    Builds a 2-stage FCFF DCF valuation from a base revenue figure, ML
    driver forecasts (revenue growth, EBIT margin, CapEx % of revenue),
    a WACC estimate, and a terminal growth assumption.

    Parameters
    ----------
    terminal_growth_rate : float, default 0.025
        Perpetuity growth rate applied in the Gordon Growth terminal value.
        Should not exceed long-run nominal GDP growth (~2-3%).
    """

    def __init__(self, terminal_growth_rate: float = 0.025) -> None:
        self.g = terminal_growth_rate

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def project_fcff(
        self,
        base_revenue: float,
        revenue_growth: np.ndarray,
        ebit_margin: np.ndarray,
        capex_pct_revenue: np.ndarray,
        tax_rate: float,
        da_pct_revenue: float = 0.03,
        nwc_pct_revenue_change: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Rolls forward revenue using ML-forecast growth rates and derives
        FCFF for each explicit forecast year.

        Parameters
        ----------
        base_revenue : float
            Most recent actual (trailing) annual revenue.
        revenue_growth, ebit_margin, capex_pct_revenue : np.ndarray
            Explicit-year ML forecasts (decimals), length = horizon.
        tax_rate : float
            Effective/marginal tax rate applied to EBIT.
        da_pct_revenue : float, default 0.03
            D&A as % of revenue (held constant absent an explicit forecast).
        nwc_pct_revenue_change : float, default 0.01
            Incremental NWC investment as % of the *change* in revenue.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (projected_revenue, projected_fcff), both length = horizon.
        """
        horizon = len(revenue_growth)
        revenue = np.zeros(horizon)
        fcff = np.zeros(horizon)

        prior_revenue = base_revenue
        for t in range(horizon):
            revenue[t] = prior_revenue * (1 + revenue_growth[t])
            ebit = revenue[t] * ebit_margin[t]
            nopat = ebit * (1 - tax_rate)
            da = revenue[t] * da_pct_revenue
            capex = revenue[t] * capex_pct_revenue[t]
            delta_revenue = revenue[t] - prior_revenue
            delta_nwc = delta_revenue * nwc_pct_revenue_change

            fcff[t] = nopat + da - capex - delta_nwc
            prior_revenue = revenue[t]

        return revenue, fcff

    def discount_cash_flows(self, fcff: np.ndarray, wacc: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Discounts each explicit-year FCFF back to present value using
        mid-year-convention-free (end-of-year) discounting.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (discount_factors, pv_fcff)
        """
        t = np.arange(1, len(fcff) + 1)
        discount_factors = 1.0 / (1 + wacc) ** t
        pv_fcff = fcff * discount_factors
        return discount_factors, pv_fcff

    def compute_terminal_value(self, terminal_year_fcff: float, wacc: float) -> float:
        """Gordon Growth Model: TV_n = FCFF_n * (1+g) / (WACC - g)."""
        if wacc <= self.g:
            logger.warning(
                "WACC (%.2f%%) <= terminal growth (%.2f%%); clamping WACC-g spread to 1%%.",
                wacc * 100, self.g * 100,
            )
            spread = 0.01
        else:
            spread = wacc - self.g
        return terminal_year_fcff * (1 + self.g) / spread

    def run(
        self,
        base_revenue: float,
        revenue_growth: np.ndarray,
        ebit_margin: np.ndarray,
        capex_pct_revenue: np.ndarray,
        tax_rate: float,
        wacc: float,
        total_debt: float,
        cash_and_equivalents: float,
        diluted_shares: float,
        current_price: float,
        forecast_years: list[int],
        da_pct_revenue: float = 0.03,
        nwc_pct_revenue_change: float = 0.01,
    ) -> DCFResult:
        """
        Executes the full 2-stage DCF and returns a fully populated
        DCFResult with every intermediate figure needed for the dashboard.
        """
        projected_revenue, projected_fcff = self.project_fcff(
            base_revenue, revenue_growth, ebit_margin, capex_pct_revenue,
            tax_rate, da_pct_revenue, nwc_pct_revenue_change,
        )

        discount_factors, pv_fcff = self.discount_cash_flows(projected_fcff, wacc)

        terminal_value = self.compute_terminal_value(projected_fcff[-1], wacc)
        pv_terminal_value = terminal_value * discount_factors[-1]

        enterprise_value = float(np.sum(pv_fcff) + pv_terminal_value)
        net_debt = total_debt - cash_and_equivalents
        equity_value = enterprise_value - net_debt
        implied_price = equity_value / diluted_shares if diluted_shares else np.nan

        upside = (
            (implied_price - current_price) / current_price * 100
            if current_price and not np.isnan(current_price) and current_price != 0
            else np.nan
        )

        result = DCFResult(
            projection_years=forecast_years,
            projected_revenue=projected_revenue,
            projected_fcff=projected_fcff,
            discount_factors=discount_factors,
            pv_fcff=pv_fcff,
            terminal_value=terminal_value,
            pv_terminal_value=pv_terminal_value,
            enterprise_value=enterprise_value,
            net_debt=net_debt,
            equity_value=equity_value,
            diluted_shares=diluted_shares,
            implied_share_price=implied_price,
            current_price=current_price,
            upside_downside_pct=upside,
            wacc=wacc,
            terminal_growth_rate=self.g,
        )
        logger.info(
            "EV=$%.1fM | Equity Value=$%.1fM | Implied Price=$%.2f | Current Price=$%.2f | Upside=%.1f%%",
            enterprise_value / 1e6, equity_value / 1e6, implied_price, current_price, upside,
        )
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data_fetcher import DataFetcher
    from ml_forecaster import MLForecaster
    from wacc_engine import WACCEngine
    from data_fetcher import fetch_market_series

    ticker = "AAPL"
    fetcher = DataFetcher(ticker)
    financials = fetcher.fetch_all()

    forecaster = MLForecaster()
    table = forecaster.build_training_table(financials.income_statement)
    revenue_row = MLForecaster._row(financials.income_statement, ["Total Revenue", "Operating Revenue"])
    table = forecaster.attach_capex_ratio(table, financials.cash_flow, revenue_row)
    forecast = forecaster.fit_and_forecast(table)

    stock_px = fetch_market_series(ticker)
    mkt_px = fetch_market_series("^GSPC")
    tnx = fetch_market_series("^TNX")
    wacc_engine = WACCEngine()
    wacc_result = wacc_engine.run(
        stock_prices=stock_px,
        market_prices=mkt_px,
        treasury_yield_series=tnx,
        market_cap=financials.info.get("marketCap", np.nan),
        total_debt=financials.total_debt,
        interest_expense=0.0,
        tax_rate=financials.effective_tax_rate,
    )

    dcf = DCFCalculator(terminal_growth_rate=0.025)
    base_rev = float(revenue_row.iloc[-1])
    result = dcf.run(
        base_revenue=base_rev,
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
    print(result.summary_frame())
    print(f"Implied Share Price: ${result.implied_share_price:.2f}")
