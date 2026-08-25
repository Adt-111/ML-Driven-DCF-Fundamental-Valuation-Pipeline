"""
Sensitivity analysis and Monte Carlo simulation over the DCF's key
uncertain inputs: revenue growth, EBIT margin, WACC, and terminal growth
rate. Produces a distribution of implied share prices to quantify
valuation uncertainty, plus a 2D sensitivity (data-table) grid of
WACC vs. terminal growth rate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dcf_calculator import DCFCalculator

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Simulated implied-price distribution and summary statistics."""

    simulated_prices: np.ndarray
    mean_price: float
    median_price: float
    std_dev: float
    p5: float
    p25: float
    p75: float
    p95: float
    prob_upside: float  # P(implied price > current price)

    def summary(self) -> dict:
        return {
            "Mean": self.mean_price,
            "Median": self.median_price,
            "Std Dev": self.std_dev,
            "5th Percentile": self.p5,
            "25th Percentile": self.p25,
            "75th Percentile": self.p75,
            "95th Percentile": self.p95,
            "P(Upside)": self.prob_upside,
        }


class MonteCarloSimulator:
    """
    Runs a Monte Carlo simulation over DCF inputs by perturbing ML-forecast
    drivers, WACC, and terminal growth with random draws, re-running the
    full DCF each iteration.

    Parameters
    ----------
    n_simulations : int, default 5000
        Number of Monte Carlo iterations.
    random_state : int, default 42
        Reproducibility seed.
    """

    def __init__(self, n_simulations: int = 5000, random_state: int = 42) -> None:
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
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
        terminal_growth_rate: float = 0.025,
        revenue_growth_std: float = 0.03,
        ebit_margin_std: float = 0.02,
        wacc_std: float = 0.01,
        terminal_growth_std: float = 0.005,
    ) -> MonteCarloResult:
        """
        Executes the Monte Carlo simulation.

        Each iteration draws:
          - revenue_growth path  ~ Normal(forecast, revenue_growth_std)
          - ebit_margin path     ~ Normal(forecast, ebit_margin_std)
          - wacc                 ~ Normal(wacc, wacc_std), floored > 0
          - terminal_growth_rate ~ Normal(g, terminal_growth_std), capped < wacc

        Returns
        -------
        MonteCarloResult
            Distribution of implied share prices with summary statistics.
        """
        horizon = len(revenue_growth)
        simulated_prices = np.full(self.n_simulations, np.nan)

        for i in range(self.n_simulations):
            sim_growth = revenue_growth + self.rng.normal(0, revenue_growth_std, horizon)
            sim_margin = np.clip(
                ebit_margin + self.rng.normal(0, ebit_margin_std, horizon), -0.5, 0.7
            )
            sim_wacc = max(0.01, wacc + self.rng.normal(0, wacc_std))
            sim_g = min(sim_wacc - 0.005, terminal_growth_rate + self.rng.normal(0, terminal_growth_std))

            calc = DCFCalculator(terminal_growth_rate=sim_g)
            try:
                result = calc.run(
                    base_revenue=base_revenue,
                    revenue_growth=sim_growth,
                    ebit_margin=sim_margin,
                    capex_pct_revenue=capex_pct_revenue,
                    tax_rate=tax_rate,
                    wacc=sim_wacc,
                    total_debt=total_debt,
                    cash_and_equivalents=cash_and_equivalents,
                    diluted_shares=diluted_shares,
                    current_price=current_price,
                    forecast_years=forecast_years,
                )
                simulated_prices[i] = result.implied_share_price
            except Exception as exc:  # noqa: BLE001
                logger.debug("Simulation iteration %d failed: %s", i, exc)
                continue

        simulated_prices = simulated_prices[~np.isnan(simulated_prices)]
        simulated_prices = simulated_prices[simulated_prices > 0]  # discard degenerate draws

        if len(simulated_prices) == 0:
            logger.error("Monte Carlo produced no valid simulations.")
            return MonteCarloResult(
                simulated_prices=np.array([]),
                mean_price=np.nan, median_price=np.nan, std_dev=np.nan,
                p5=np.nan, p25=np.nan, p75=np.nan, p95=np.nan, prob_upside=np.nan,
            )

        prob_upside = float(np.mean(simulated_prices > current_price)) if current_price else np.nan

        result = MonteCarloResult(
            simulated_prices=simulated_prices,
            mean_price=float(np.mean(simulated_prices)),
            median_price=float(np.median(simulated_prices)),
            std_dev=float(np.std(simulated_prices)),
            p5=float(np.percentile(simulated_prices, 5)),
            p25=float(np.percentile(simulated_prices, 25)),
            p75=float(np.percentile(simulated_prices, 75)),
            p95=float(np.percentile(simulated_prices, 95)),
            prob_upside=prob_upside,
        )
        logger.info(
            "Monte Carlo (%d valid sims): mean=$%.2f median=$%.2f std=$%.2f P(upside)=%.1f%%",
            len(simulated_prices), result.mean_price, result.median_price,
            result.std_dev, result.prob_upside * 100,
        )
        return result

    def sensitivity_grid(
        self,
        base_revenue: float,
        revenue_growth: np.ndarray,
        ebit_margin: np.ndarray,
        capex_pct_revenue: np.ndarray,
        tax_rate: float,
        total_debt: float,
        cash_and_equivalents: float,
        diluted_shares: float,
        current_price: float,
        forecast_years: list[int],
        wacc_center: float,
        terminal_growth_center: float = 0.025,
        wacc_range: tuple[float, float] = (-0.02, 0.02),
        growth_range: tuple[float, float] = (-0.01, 0.01),
        steps: int = 5,
    ) -> pd.DataFrame:
        """
        Builds a 2D data-table of implied share price across a grid of
        WACC (rows) x Terminal Growth Rate (columns), holding all other
        drivers at their point-estimate forecast.

        Returns
        -------
        pd.DataFrame
            Index = WACC values, Columns = terminal growth values, cells =
            implied share price.
        """
        wacc_vals = np.linspace(wacc_center + wacc_range[0], wacc_center + wacc_range[1], steps)
        growth_vals = np.linspace(
            terminal_growth_center + growth_range[0], terminal_growth_center + growth_range[1], steps
        )

        grid = pd.DataFrame(
            index=[f"{w*100:.2f}%" for w in wacc_vals],
            columns=[f"{g*100:.2f}%" for g in growth_vals],
            dtype=float,
        )

        for w in wacc_vals:
            for g in growth_vals:
                if w <= g:
                    continue
                calc = DCFCalculator(terminal_growth_rate=g)
                result = calc.run(
                    base_revenue=base_revenue,
                    revenue_growth=revenue_growth,
                    ebit_margin=ebit_margin,
                    capex_pct_revenue=capex_pct_revenue,
                    tax_rate=tax_rate,
                    wacc=w,
                    total_debt=total_debt,
                    cash_and_equivalents=cash_and_equivalents,
                    diluted_shares=diluted_shares,
                    current_price=current_price,
                    forecast_years=forecast_years,
                )
                grid.loc[f"{w*100:.2f}%", f"{g*100:.2f}%"] = result.implied_share_price

        return grid


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data_fetcher import DataFetcher, fetch_market_series
    from ml_forecaster import MLForecaster
    from wacc_engine import WACCEngine

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
        stock_prices=stock_px, market_prices=mkt_px, treasury_yield_series=tnx,
        market_cap=financials.info.get("marketCap", np.nan),
        total_debt=financials.total_debt, interest_expense=0.0,
        tax_rate=financials.effective_tax_rate,
    )

    base_rev = float(revenue_row.iloc[-1])
    mc = MonteCarloSimulator(n_simulations=2000)
    mc_result = mc.run(
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
    print(mc_result.summary())
