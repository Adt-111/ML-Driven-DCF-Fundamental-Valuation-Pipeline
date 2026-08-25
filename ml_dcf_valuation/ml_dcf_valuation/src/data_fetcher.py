"""
Retrieves raw fundamental and market data for a given equity ticker using
`yfinance`, normalizes the annual Income Statement / Balance Sheet / Cash
Flow Statement, and derives historical Unlevered Free Cash Flow to the Firm
(FCFF).

FCFF = EBIT * (1 - tax_rate) + D&A - CapEx - Delta_NWC

All public functions are typed and documented so the module can be unit
tested or swapped for an FMP (Financial Modeling Prep) API backend later
without changing the downstream interface (see `FMP_STUB` at bottom).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class CompanyFinancials:
    """Container for all raw + derived data pulled for a single ticker."""

    ticker: str
    income_statement: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance_sheet: pd.DataFrame = field(default_factory=pd.DataFrame)
    cash_flow: pd.DataFrame = field(default_factory=pd.DataFrame)
    fcff_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    info: dict = field(default_factory=dict)
    shares_outstanding: float = np.nan
    current_price: float = np.nan
    total_debt: float = np.nan
    cash_and_equivalents: float = np.nan
    effective_tax_rate: float = np.nan


class DataFetcher:
    """
    Pulls and cleans fundamental statements for a ticker via yfinance and
    computes a historical FCFF time series.

    Parameters
    ----------
    ticker : str
        Equity ticker symbol, e.g. "AAPL".
    lookback_years : int, default 5
        Number of most-recent annual periods to retain.
    """

    def __init__(self, ticker: str, lookback_years: int = 5) -> None:
        self.ticker = ticker.upper().strip()
        self.lookback_years = lookback_years
        self._yf_ticker = yf.Ticker(self.ticker)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fetch_all(self) -> CompanyFinancials:
        """
        Orchestrates the full fetch + derive pipeline.

        Returns
        -------
        CompanyFinancials
            Populated dataclass with statements, market data, and FCFF history.
        """
        logger.info("Fetching statements for %s ...", self.ticker)

        income_stmt = self._safe_fetch(lambda: self._yf_ticker.income_stmt, "income_stmt")
        balance_sheet = self._safe_fetch(lambda: self._yf_ticker.balance_sheet, "balance_sheet")
        cash_flow = self._safe_fetch(lambda: self._yf_ticker.cash_flow, "cash_flow")
        info = self._safe_fetch(lambda: self._yf_ticker.get_info(), "info", default={})

        income_stmt = self._trim_and_transpose(income_stmt)
        balance_sheet = self._trim_and_transpose(balance_sheet)
        cash_flow = self._trim_and_transpose(cash_flow)

        shares_out = self._extract_shares_outstanding(info, balance_sheet)
        current_price = float(info.get("currentPrice") or info.get("regularMarketPrice") or np.nan)
        total_debt = self._extract_total_debt(balance_sheet, info)
        cash_eq = self._extract_cash(balance_sheet, info)
        tax_rate = self._estimate_effective_tax_rate(income_stmt)

        fcff_history = self.compute_historical_fcff(income_stmt, cash_flow, tax_rate)

        financials = CompanyFinancials(
            ticker=self.ticker,
            income_statement=income_stmt,
            balance_sheet=balance_sheet,
            cash_flow=cash_flow,
            fcff_history=fcff_history,
            info=info,
            shares_outstanding=shares_out,
            current_price=current_price,
            total_debt=total_debt,
            cash_and_equivalents=cash_eq,
            effective_tax_rate=tax_rate,
        )
        logger.info(
            "Fetch complete for %s | periods=%d | tax_rate=%.2f%% | shares_out=%.0f",
            self.ticker, len(fcff_history), tax_rate * 100, shares_out,
        )
        return financials

    def compute_historical_fcff(
        self,
        income_stmt: pd.DataFrame,
        cash_flow: pd.DataFrame,
        tax_rate: float,
    ) -> pd.DataFrame:
        """
        Derives FCFF = EBIT*(1-t) + D&A - CapEx - Delta_NWC for each period.

        Notes
        -----
        Delta_NWC is approximated from the cash-flow statement's
        "Change In Working Capital" line when available; otherwise it
        defaults to 0 for that period (flagged in logs).
        """
        if income_stmt.empty or cash_flow.empty:
            logger.warning("Insufficient data to compute FCFF history for %s", self.ticker)
            return pd.DataFrame()

        ebit = self._first_available(income_stmt, ["EBIT", "Operating Income"])
        d_and_a = self._first_available(
            cash_flow, ["Depreciation And Amortization", "Depreciation Amortization Depletion"]
        )
        capex = self._first_available(cash_flow, ["Capital Expenditure", "Purchase Of PPE"])
        delta_nwc = self._first_available(cash_flow, ["Change In Working Capital"])

        idx = ebit.index.intersection(d_and_a.index).intersection(capex.index)
        rows = []
        for period in idx:
            ebit_v = ebit.get(period, np.nan)
            da_v = d_and_a.get(period, np.nan)
            capex_v = capex.get(period, np.nan)  # yfinance reports CapEx as negative outflow
            nwc_v = delta_nwc.get(period, 0.0) if delta_nwc is not None else 0.0

            nopat = ebit_v * (1 - tax_rate)
            fcff = nopat + da_v + capex_v - nwc_v  # capex_v already negative -> subtracts
            rows.append(
                {
                    "period": period,
                    "EBIT": ebit_v,
                    "D&A": da_v,
                    "CapEx": capex_v,
                    "Delta_NWC": nwc_v,
                    "NOPAT": nopat,
                    "FCFF": fcff,
                }
            )
        df = pd.DataFrame(rows).sort_values("period").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _safe_fetch(self, fn, name: str, default=None):
        try:
            result = fn()
            if result is None or (hasattr(result, "empty") and result.empty):
                logger.warning("%s returned empty for %s", name, self.ticker)
                return default if default is not None else pd.DataFrame()
            return result
        except Exception as exc:  # noqa: BLE001 - yfinance raises assorted errors
            logger.error("Failed to fetch %s for %s: %s", name, self.ticker, exc)
            return default if default is not None else pd.DataFrame()

    def _trim_and_transpose(self, df: pd.DataFrame) -> pd.DataFrame:
        """yfinance returns statements as (line_item x period); we keep that
        orientation but trim to `lookback_years` most-recent columns and
        ensure column headers are sorted oldest -> newest."""
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.sort_index(axis=1)
        cols = df.columns[-self.lookback_years:]
        return df[cols]

    @staticmethod
    def _first_available(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
        for name in candidates:
            if name in df.index:
                return df.loc[name]
        return pd.Series(dtype=float)

    @staticmethod
    def _extract_shares_outstanding(info: dict, balance_sheet: pd.DataFrame) -> float:
        shares = info.get("sharesOutstanding")
        if shares:
            return float(shares)
        for name in ["Diluted Average Shares", "Ordinary Shares Number"]:
            if name in balance_sheet.index:
                return float(balance_sheet.loc[name].iloc[-1])
        return np.nan

    @staticmethod
    def _extract_total_debt(balance_sheet: pd.DataFrame, info: dict) -> float:
        if "Total Debt" in balance_sheet.index:
            return float(balance_sheet.loc["Total Debt"].iloc[-1])
        debt = info.get("totalDebt")
        return float(debt) if debt else 0.0

    @staticmethod
    def _extract_cash(balance_sheet: pd.DataFrame, info: dict) -> float:
        for name in ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]:
            if name in balance_sheet.index:
                return float(balance_sheet.loc[name].iloc[-1])
        cash = info.get("totalCash")
        return float(cash) if cash else 0.0

    @staticmethod
    def _estimate_effective_tax_rate(income_stmt: pd.DataFrame, default: float = 0.21) -> float:
        try:
            tax = DataFetcher._first_available(income_stmt, ["Tax Provision"])
            pretax = DataFetcher._first_available(income_stmt, ["Pretax Income"])
            common = tax.index.intersection(pretax.index)
            ratios = [
                tax[p] / pretax[p]
                for p in common
                if pretax[p] not in (0, np.nan) and not np.isnan(pretax[p])
            ]
            ratios = [r for r in ratios if 0.0 <= r <= 0.5]
            if ratios:
                return float(np.mean(ratios))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tax rate estimation failed, using default: %s", exc)
        return default


def fetch_market_series(symbol: str, period: str = "5y", interval: str = "1mo") -> pd.Series:
    """
    Fetches an adjusted-close price series for a market index/rate proxy
    (e.g. ^GSPC, ^TNX) used by the WACC engine.

    Parameters
    ----------
    symbol : str
        Yahoo Finance symbol, e.g. "^GSPC" (S&P 500) or "^TNX" (10Y yield).
    period : str
        yfinance lookback period string.
    interval : str
        Sampling interval, e.g. "1mo" for monthly returns.

    Returns
    -------
    pd.Series
        Close price series indexed by date.
    """
    try:
        data = yf.Ticker(symbol).history(period=period, interval=interval)
        if data.empty:
            logger.warning("No market data returned for %s", symbol)
            return pd.Series(dtype=float)
        return data["Close"].dropna()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fetch market series for %s: %s", symbol, exc)
        return pd.Series(dtype=float)


# fmp_stub: alternative backend (financial modeling prep api)
class FMPDataFetcher:
    """
    Stub adapter for Financial Modeling Prep (FMP) as an alternative data
    source. Implements the same public interface as `DataFetcher` so it can
    be swapped in via dependency injection. Requires an FMP_API_KEY.

    Not wired into the default pipeline (yfinance is used out-of-the-box to
    keep the project runnable with no paid API key), but included so
    production deployments can switch providers without touching
    `ml_forecaster.py`, `dcf_calculator.py`, or `app.py`.
    """

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, ticker: str, api_key: Optional[str] = None) -> None:
        self.ticker = ticker.upper().strip()
        self.api_key = api_key
        if not api_key:
            logger.warning("FMPDataFetcher instantiated without an API key; calls will fail.")

    def fetch_all(self) -> CompanyFinancials:
        raise NotImplementedError(
            "FMP integration is a stub. Implement HTTP calls to "
            f"{self.BASE_URL}/income-statement/{self.ticker}?apikey=... "
            "and map fields to CompanyFinancials to activate."
        )


if __name__ == "__main__":
    fetcher = DataFetcher("AAPL")
    data = fetcher.fetch_all()
    print(data.fcff_history)
