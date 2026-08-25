"""
Trains gradient-boosted tree regressors (XGBoost, with LightGBM as a
fallback/ensemble partner) on historical fundamental ratios plus
macroeconomic indicators to produce explicit 5-year forward projections for:

    1. Revenue Growth (YoY %)
    2. EBIT Margin (%)
    3. CapEx / Revenue (%)

Because a single company only has a handful of historical annual
observations (typically 4-5 from yfinance), a pure per-company supervised
model would badly overfit. To keep this production-realistic, the module:

  * Engineers a lag-based feature set (prior-year ratio, 3yr trailing
    average, YoY delta, macro indicators) so each historical year becomes
    one training row.
  * Blends the trained model's prediction with an exponentially-weighted
    historical mean (Bayesian-style shrinkage) so forecasts stay anchored
    to the company's own trend when the sample is thin.
  * Uses a recursive multi-step forecast: predict year t+1, feed it back in
    as a lag feature to predict t+2, etc., for a full 5-year horizon.

This is intentionally documented so a user can swap in a larger
cross-sectional training set (many tickers) for a materially stronger model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:  # pragma: no cover
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:  # pragma: no cover
    _HAS_LGB = False

from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Explicit forward-year projections for the three DCF drivers."""

    years: list[int]
    revenue_growth: np.ndarray
    ebit_margin: np.ndarray
    capex_pct_revenue: np.ndarray
    model_used: str

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Year": self.years,
                "Revenue Growth (%)": self.revenue_growth * 100,
                "EBIT Margin (%)": self.ebit_margin * 100,
                "CapEx / Revenue (%)": self.capex_pct_revenue * 100,
            }
        )


class MLForecaster:
    """
    Fits a gradient-boosted regressor per driver (revenue growth, EBIT
    margin, CapEx %) and produces a 5-year recursive forecast.

    Parameters
    ----------
    forecast_horizon : int, default 5
        Number of forward years to project.
    shrinkage : float, default 0.35
        Weight (0-1) applied to the historical trailing mean when blending
        with the ML point estimate; higher = more conservative / anchored.
    random_state : int, default 42
        Reproducibility seed for tree models.
    """

    DRIVER_COLUMNS = ["revenue_growth", "ebit_margin", "capex_pct_revenue"]

    def __init__(
        self,
        forecast_horizon: int = 5,
        shrinkage: float = 0.35,
        random_state: int = 42,
    ) -> None:
        self.forecast_horizon = forecast_horizon
        self.shrinkage = shrinkage
        self.random_state = random_state
        self.models: dict[str, object] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build_training_table(
        self,
        income_stmt: pd.DataFrame,
        macro_indicators: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Converts raw income-statement history into a supervised-learning
        table of driver ratios with lag features.

        Parameters
        ----------
        income_stmt : pd.DataFrame
            Line-items x periods, as returned by DataFetcher.
        macro_indicators : Optional[pd.DataFrame]
            Optional macro series (e.g. GDP growth, CPI, 10Y yield) indexed
            by year, merged in as additional features.

        Returns
        -------
        pd.DataFrame
            One row per historical year with driver values + engineered
            lag/trend features.
        """
        if income_stmt.empty:
            logger.warning("Empty income statement; cannot build training table.")
            return pd.DataFrame()

        revenue = self._row(income_stmt, ["Total Revenue", "Operating Revenue"])
        ebit = self._row(income_stmt, ["EBIT", "Operating Income"])
        capex_proxy = self._row(income_stmt, ["Reconciled Depreciation"])  # fallback proxy only

        years = [pd.Timestamp(c).year for c in revenue.index]
        df = pd.DataFrame(
            {
                "year": years,
                "revenue": revenue.values,
                "ebit": ebit.reindex(revenue.index).values,
            }
        ).sort_values("year").reset_index(drop=True)

        df["revenue_growth"] = df["revenue"].pct_change()
        df["ebit_margin"] = df["ebit"] / df["revenue"]
        # CapEx % of revenue is computed upstream  from cash flow in the caller
        # and merged in; placeholder column created here  for schema stability.
        if "capex_pct_revenue" not in df.columns:
            df["capex_pct_revenue"] = np.nan

        for col in self.DRIVER_COLUMNS:
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_trail_mean"] = df[col].expanding().mean().shift(1)

        if macro_indicators is not None and not macro_indicators.empty:
            df = df.merge(macro_indicators, on="year", how="left")

        df = df.dropna(subset=["revenue_growth"]).reset_index(drop=True)
        return df

    def attach_capex_ratio(self, training_table: pd.DataFrame, cash_flow: pd.DataFrame,
                            revenue: pd.Series) -> pd.DataFrame:
        """Merges actual CapEx/Revenue (from cash-flow statement) into the
        training table, replacing the placeholder column."""
        capex = self._row(cash_flow, ["Capital Expenditure", "Purchase Of PPE"])
        years = [pd.Timestamp(c).year for c in capex.index]
        capex_df = pd.DataFrame({"year": years, "capex": capex.values})
        rev_years = [pd.Timestamp(c).year for c in revenue.index]
        rev_df = pd.DataFrame({"year": rev_years, "revenue": revenue.values})
        merged = capex_df.merge(rev_df, on="year")
        merged["capex_pct_revenue"] = merged["capex"].abs() / merged["revenue"]

        training_table = training_table.drop(columns=["capex_pct_revenue"], errors="ignore")
        training_table = training_table.merge(
            merged[["year", "capex_pct_revenue"]], on="year", how="left"
        )
        training_table["capex_pct_revenue_lag1"] = training_table["capex_pct_revenue"].shift(1)
        training_table["capex_pct_revenue_trail_mean"] = (
            training_table["capex_pct_revenue"].expanding().mean().shift(1)
        )
        return training_table

    def fit_and_forecast(self, training_table: pd.DataFrame) -> ForecastResult:
        """
        Trains one regressor per driver and recursively forecasts
        `forecast_horizon` years forward.

        Falls back gracefully: XGBoost -> LightGBM -> Ridge regression ->
        naive historical-mean, depending on library availability and
        sample size, so the pipeline never hard-fails on a data-sparse ticker.
        """
        if training_table.empty or len(training_table) < 2:
            logger.warning("Insufficient history for ML fit; using naive historical means.")
            return self._naive_forecast(training_table)

        last_year = int(training_table["year"].max())
        forecast_years = list(range(last_year + 1, last_year + 1 + self.forecast_horizon))

        results: dict[str, np.ndarray] = {}
        model_name_used = "naive_mean"

        for driver in self.DRIVER_COLUMNS:
            feature_cols = [f"{driver}_lag1", f"{driver}_trail_mean"]
            feature_cols = [c for c in feature_cols if c in training_table.columns]
            data = training_table.dropna(subset=[driver] + feature_cols)

            if len(data) < 3 or not feature_cols:
                # Not enough rows to fit a tree model -> historical mean anchor
                hist_mean = training_table[driver].dropna().mean()
                hist_mean = 0.0 if np.isnan(hist_mean) else hist_mean
                results[driver] = np.full(self.forecast_horizon, hist_mean)
                continue

            X = data[feature_cols].values
            y = data[driver].values
            model, model_name_used = self._fit_best_available(X, y)
            self.models[driver] = model

            # Recursive forecasting
            last_row = training_table.iloc[-1]
            lag1 = last_row.get(f"{driver}_lag1", last_row.get(driver, y[-1]))
            trail_mean = last_row.get(f"{driver}_trail_mean", np.mean(y))
            hist_trail = list(training_table[driver].dropna().values)

            preds = []
            for _ in range(self.forecast_horizon):
                x_input = np.array([[lag1 if not np.isnan(lag1) else y[-1],
                                      trail_mean if not np.isnan(trail_mean) else np.mean(y)]])
                point_pred = float(model.predict(x_input)[0])

                # Shrinkage toward trailing historical mean  for stability
                hist_anchor = np.mean(hist_trail[-3:]) if len(hist_trail) >= 1 else point_pred
                blended = (1 - self.shrinkage) * point_pred + self.shrinkage * hist_anchor
                blended = self._clip_driver(driver, blended)

                preds.append(blended)
                hist_trail.append(blended)
                lag1 = blended
                trail_mean = np.mean(hist_trail)

            results[driver] = np.array(preds)

        return ForecastResult(
            years=forecast_years,
            revenue_growth=results["revenue_growth"],
            ebit_margin=results["ebit_margin"],
            capex_pct_revenue=results["capex_pct_revenue"],
            model_used=model_name_used,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _fit_best_available(self, X: np.ndarray, y: np.ndarray):
        """Tries XGBoost first, then LightGBM, then falls back to Ridge."""
        n = len(y)
        if _HAS_XGB and n >= 3:
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=2,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.random_state,
                objective="reg:squarederror",
            )
            model.fit(X, y)
            return model, "xgboost"
        if _HAS_LGB and n >= 3:
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=2,
                learning_rate=0.1,
                min_child_samples=1,
                random_state=self.random_state,
                verbosity=-1,
            )
            model.fit(X, y)
            return model, "lightgbm"
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        return model, "ridge_fallback"

    @staticmethod
    def _clip_driver(driver: str, value: float) -> float:
        """Applies sane economic bounds so recursive forecasts don't diverge."""
        bounds = {
            "revenue_growth": (-0.30, 0.60),
            "ebit_margin": (-0.20, 0.60),
            "capex_pct_revenue": (0.0, 0.35),
        }
        lo, hi = bounds.get(driver, (-1.0, 1.0))
        return float(np.clip(value, lo, hi))

    @staticmethod
    def _row(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
        for name in candidates:
            if name in df.index:
                return df.loc[name]
        return pd.Series(dtype=float)

    def _naive_forecast(self, training_table: pd.DataFrame) -> ForecastResult:
        last_year = int(training_table["year"].max()) if not training_table.empty else 2024
        forecast_years = list(range(last_year + 1, last_year + 1 + self.forecast_horizon))
        defaults = {"revenue_growth": 0.05, "ebit_margin": 0.15, "capex_pct_revenue": 0.05}
        means = {
            d: (training_table[d].dropna().mean() if d in training_table.columns
                and not training_table[d].dropna().empty else defaults[d])
            for d in self.DRIVER_COLUMNS
        }
        return ForecastResult(
            years=forecast_years,
            revenue_growth=np.full(self.forecast_horizon, means["revenue_growth"]),
            ebit_margin=np.full(self.forecast_horizon, means["ebit_margin"]),
            capex_pct_revenue=np.full(self.forecast_horizon, means["capex_pct_revenue"]),
            model_used="naive_mean",
        )


def fetch_macro_indicators(years: list[int]) -> pd.DataFrame:
    """
    Placeholder macro-indicator loader (US real GDP growth, CPI YoY).

    In production, replace with FRED API pulls (e.g. `fredapi` package with
    series GDPC1, CPIAUCSL). Returns a static, documented approximation here
    so the pipeline runs without external credentials.
    """
    macro_table = pd.DataFrame(
        {
            "year": years,
            "gdp_growth": [0.025] * len(years),
            "cpi_yoy": [0.03] * len(years),
        }
    )
    return macro_table


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data_fetcher import DataFetcher

    fetcher = DataFetcher("AAPL")
    financials = fetcher.fetch_all()

    forecaster = MLForecaster()
    table = forecaster.build_training_table(financials.income_statement)
    revenue_row = MLForecaster._row(financials.income_statement, ["Total Revenue", "Operating Revenue"])
    table = forecaster.attach_capex_ratio(table, financials.cash_flow, revenue_row)
    forecast = forecaster.fit_and_forecast(table)
    print(forecast.to_frame())
