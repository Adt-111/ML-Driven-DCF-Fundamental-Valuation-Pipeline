"""
Computes the Weighted Average Cost of Capital (WACC) used to discount
projected FCFF in the DCF model.

Components
----------
1. Cost of Equity (r_e) via CAPM:
       r_e = R_f + Beta * ERP
   - R_f  : 10-Year U.S. Treasury yield (^TNX), most recent observation.
   - Beta : Rolling 36-month OLS regression of monthly stock returns on
            S&P 500 (^GSPC) monthly returns.
   - ERP  : Equity Risk Premium (configurable, default 5.0%).

2. Cost of Debt (r_d):
       r_d = Interest Expense / Total Debt   (effective rate proxy)
   Falls back to a credit-spread-over-risk-free heuristic if interest
   expense / debt are unavailable.

3. WACC:
       WACC = (E/V) * r_e + (D/V) * r_d * (1 - t)
   where E = market cap, D = total debt, V = E + D.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WACCResult:
    """Holds every intermediate output for transparency/auditability."""

    beta: float
    risk_free_rate: float
    equity_risk_premium: float
    cost_of_equity: float
    cost_of_debt: float
    tax_rate: float
    market_cap: float
    total_debt: float
    weight_equity: float
    weight_debt: float
    wacc: float


class WACCEngine:
    """
    Computes WACC for a given ticker using rolling-beta CAPM and a
    debt/equity capital-structure weighting.

    Parameters
    ----------
    equity_risk_premium : float, default 0.05
        Long-run assumed ERP (5.0%). Configurable per macro regime.
    beta_window_months : int, default 36
        Rolling window length (in months) for the beta regression.
    """

    def __init__(self, equity_risk_premium: float = 0.05, beta_window_months: int = 36) -> None:
        self.erp = equity_risk_premium
        self.beta_window = beta_window_months

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def compute_rolling_beta(
        self,
        stock_prices: pd.Series,
        market_prices: pd.Series,
    ) -> float:
        """
        Runs an OLS regression of stock monthly returns on S&P 500 monthly
        returns over the most recent `beta_window_months` observations.

        beta = Cov(r_stock, r_market) / Var(r_market)

        Returns
        -------
        float
            Estimated beta. Falls back to 1.0 (market beta) if insufficient
            overlapping data exists.
        """
        if stock_prices.empty or market_prices.empty:
            logger.warning("Missing price series; defaulting beta to 1.0")
            return 1.0

        stock_ret = stock_prices.pct_change().dropna()
        market_ret = market_prices.pct_change().dropna()

        aligned = pd.concat([stock_ret, market_ret], axis=1, join="inner").dropna()
        aligned.columns = ["stock", "market"]

        if len(aligned) < 6:
            logger.warning(
                "Only %d overlapping return observations; defaulting beta to 1.0",
                len(aligned),
            )
            return 1.0

        window = aligned.tail(self.beta_window)
        covariance = np.cov(window["stock"], window["market"])[0, 1]
        variance = np.var(window["market"], ddof=1)

        if variance == 0 or np.isnan(variance):
            logger.warning("Zero market variance in window; defaulting beta to 1.0")
            return 1.0

        beta = covariance / variance
        logger.info("Computed rolling beta over %d months: %.3f", len(window), beta)
        return float(beta)

    def compute_risk_free_rate(self, treasury_yield_series: pd.Series) -> float:
        """
        Extracts the most recent 10-Year Treasury yield (^TNX quotes yield
        in percentage points x10 convention handled by yfinance as e.g.
        4.25 meaning 4.25%), converted to a decimal.

        Returns
        -------
        float
            Risk-free rate as a decimal (e.g. 0.042 for 4.2%).
        """
        if treasury_yield_series.empty:
            logger.warning("No ^TNX data; defaulting risk-free rate to 4.0%%")
            return 0.04
        latest = float(treasury_yield_series.dropna().iloc[-1])
        rf = latest / 100.0
        logger.info("Risk-free rate (10Y UST) = %.3f%%", rf * 100)
        return rf

    def compute_cost_of_equity(self, beta: float, risk_free_rate: float) -> float:
        """CAPM: r_e = R_f + Beta * ERP."""
        return risk_free_rate + beta * self.erp

    def compute_cost_of_debt(
        self,
        interest_expense: float,
        total_debt: float,
        risk_free_rate: float,
        default_spread: float = 0.02,
    ) -> float:
        """
        Effective pre-tax cost of debt = Interest Expense / Total Debt.
        Falls back to Risk-Free Rate + default credit spread if inputs are
        missing or degenerate.
        """
        if total_debt and total_debt > 0 and interest_expense and interest_expense > 0:
            r_d = interest_expense / total_debt
            if 0.0 < r_d < 0.25:  # sanity bound
                return r_d
        logger.warning(
            "Falling back to risk-free + credit spread for cost of debt (rf=%.3f, spread=%.3f)",
            risk_free_rate, default_spread,
        )
        return risk_free_rate + default_spread

    def compute_wacc(
        self,
        market_cap: float,
        total_debt: float,
        cost_of_equity: float,
        cost_of_debt: float,
        tax_rate: float,
    ) -> tuple[float, float, float]:
        """
        WACC = (E/V)*r_e + (D/V)*r_d*(1-t)

        Returns
        -------
        tuple[float, float, float]
            (wacc, weight_equity, weight_debt)
        """
        total_value = market_cap + total_debt
        if total_value <= 0:
            logger.warning("Non-positive enterprise value inputs; defaulting weights to 100%% equity")
            return cost_of_equity, 1.0, 0.0

        w_e = market_cap / total_value
        w_d = total_debt / total_value
        wacc = w_e * cost_of_equity + w_d * cost_of_debt * (1 - tax_rate)
        return wacc, w_e, w_d

    def run(
        self,
        stock_prices: pd.Series,
        market_prices: pd.Series,
        treasury_yield_series: pd.Series,
        market_cap: float,
        total_debt: float,
        interest_expense: float,
        tax_rate: float,
    ) -> WACCResult:
        """Full end-to-end WACC computation, returning an audit-friendly result object."""
        beta = self.compute_rolling_beta(stock_prices, market_prices)
        rf = self.compute_risk_free_rate(treasury_yield_series)
        r_e = self.compute_cost_of_equity(beta, rf)
        r_d = self.compute_cost_of_debt(interest_expense, total_debt, rf)
        wacc, w_e, w_d = self.compute_wacc(market_cap, total_debt, r_e, r_d, tax_rate)

        result = WACCResult(
            beta=beta,
            risk_free_rate=rf,
            equity_risk_premium=self.erp,
            cost_of_equity=r_e,
            cost_of_debt=r_d,
            tax_rate=tax_rate,
            market_cap=market_cap,
            total_debt=total_debt,
            weight_equity=w_e,
            weight_debt=w_d,
            wacc=wacc,
        )
        logger.info(
            "WACC=%.2f%% | r_e=%.2f%% | r_d=%.2f%% | beta=%.2f | W_e=%.1f%% | W_d=%.1f%%",
            wacc * 100, r_e * 100, r_d * 100, beta, w_e * 100, w_d * 100,
        )
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data_fetcher import DataFetcher, fetch_market_series

    ticker = "AAPL"
    fetcher = DataFetcher(ticker)
    financials = fetcher.fetch_all()

    stock_px = fetch_market_series(ticker)
    mkt_px = fetch_market_series("^GSPC")
    tnx = fetch_market_series("^TNX")

    market_cap = financials.info.get("marketCap", np.nan)
    interest_expense = 0.0
    if "Interest Expense" in financials.income_statement.index:
        interest_expense = float(financials.income_statement.loc["Interest Expense"].iloc[-1])

    engine = WACCEngine()
    result = engine.run(
        stock_prices=stock_px,
        market_prices=mkt_px,
        treasury_yield_series=tnx,
        market_cap=market_cap,
        total_debt=financials.total_debt,
        interest_expense=abs(interest_expense),
        tax_rate=financials.effective_tax_rate,
    )
    print(result)
