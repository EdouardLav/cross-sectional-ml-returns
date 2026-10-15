import logging
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import data_cfg

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────
# 1. S&P 100 CONSTITUENTS
# ──────────────────────────────────────────────────

def get_sp100_tickers() -> list[str]:
    """
    S&P 100 constituents (as of March 2026).
    
    Hardcoded for reproducibility and to avoid web scraping fragility.
    Survivorship bias: this is the CURRENT list, not historical constituents.
    """
    tickers = [
        "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
        "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C",
        "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
        "CVX", "DE", "DHR", "DIS", "DUK", "EMR", "EXC", "F", "FDX", "GD",
        "GE", "GILD", "GM", "GOOGL", "GS", "HD", "HON", "IBM", "INTC",
        "INTU", "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA",
        "MCD", "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT",
        "NEE", "NFLX", "NKE", "NOW", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM",
        "PYPL", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO",
        "TMUS", "TSLA", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC",
        "WMT", "XOM",
    ]
    logger.info(f"Retrieved {len(tickers)} S&P 100 tickers")
    return sorted(tickers)


# ──────────────────────────────────────────────────
# 2. PRICE & VOLUME DATA
# ──────────────────────────────────────────────────

def download_price_data(
    tickers: list[str],
    start: str = data_cfg.start_date,
    end: str = data_cfg.end_date,
) -> pd.DataFrame:
    """
    Download monthly adjusted close prices and volume from yfinance.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols (yfinance format).
    start, end : str
        Date range in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        MultiIndex columns: (field, ticker) where field ∈ {'Adj Close', 'Volume'}.
        Index: monthly DatetimeIndex.
    """
    logger.info(f"Downloading monthly data for {len(tickers)} tickers...")

    # yf.download handles batching internally with threads.
    # auto_adjust=False gives us both 'Close' and 'Adj Close'.
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1mo",
        auto_adjust=False,
        progress=True,
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("yfinance returned empty DataFrame — check your network connection")

    # Keep only Adj Close and Volume
    # yfinance >= 0.2.31 returns MultiIndex columns: (Price, Ticker)
    fields_to_keep = ["Adj Close", "Volume"]
    raw = raw[[col for col in raw.columns if col[0] in fields_to_keep]]

    logger.info(f"Downloaded data shape: {raw.shape}")
    return raw


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple monthly returns from adjusted close prices.

    Why simple returns (not log returns)?
    - Portfolio returns are weighted averages of simple returns.
    - Our evaluation builds long-short portfolios → simple returns are the
      natural choice for cross-sectional analysis.
    - Log returns are additive across time (convenient for time-series),
      but NOT across assets in a portfolio.
    """
    # Extract adjusted close prices: shape (n_months, n_tickers)
    if isinstance(prices.columns, pd.MultiIndex):
        adj_close = prices["Adj Close"]
    else:
        adj_close = prices

    returns = adj_close.pct_change()

    # Drop first row (NaN from pct_change)
    returns = returns.iloc[1:]

    return returns


def clean_returns(
    returns: pd.DataFrame,
    max_abs_ret: float = data_cfg.max_abs_return,
    min_months: int = data_cfg.min_months_per_ticker,
    min_tickers: int = data_cfg.min_tickers_per_month,
) -> pd.DataFrame:
    """
    Clean return data:
      1. Cap extreme returns (likely data errors)
      2. Drop tickers with too few observations
      3. Drop months with too few tickers

    Parameters
    ----------
    returns : pd.DataFrame
        Raw monthly returns, shape (n_months, n_tickers).
    max_abs_ret : float
        Maximum absolute return. Anything beyond is set to NaN.
    min_months : int
        Minimum number of non-NaN months per ticker.
    min_tickers : int
        Minimum number of non-NaN tickers per month.
    """
    n_before = returns.shape

    # Step 1: Flag extreme returns as NaN
    mask_extreme = returns.abs() > max_abs_ret
    n_extreme = mask_extreme.sum().sum()
    returns = returns.where(~mask_extreme)
    logger.info(f"Flagged {n_extreme} extreme returns (|r| > {max_abs_ret})")

    # Step 2: Drop tickers with too few months
    valid_months_per_ticker = returns.notna().sum(axis=0)
    valid_tickers = valid_months_per_ticker[valid_months_per_ticker >= min_months].index
    returns = returns[valid_tickers]
    logger.info(
        f"Kept {len(valid_tickers)} tickers with >= {min_months} months "
        f"(dropped {n_before[1] - len(valid_tickers)})"
    )

    # Step 3: Drop months with too few tickers
    valid_tickers_per_month = returns.notna().sum(axis=1)
    valid_months = valid_tickers_per_month[valid_tickers_per_month >= min_tickers].index
    returns = returns.loc[valid_months]
    logger.info(
        f"Kept {len(valid_months)} months with >= {min_tickers} tickers "
        f"(dropped {n_before[0] - len(valid_months)})"
    )

    return returns


def winsorize_cross_section(
    df: pd.DataFrame,
    lower: float = data_cfg.winsorize_lower,
    upper: float = data_cfg.winsorize_upper,
) -> pd.DataFrame:
    """
    Winsorize each row (cross-section) independently.

    At each date t, values below the `lower` percentile are set to that percentile,
    and values above the `upper` percentile are set to that percentile.

    Why cross-sectional (not time-series) winsorization?
    - Each month's cross-section is an independent sample for our model.
    - We want to limit the influence of outliers WITHIN a given month's ranking.
    - Time-series winsorization would inappropriately use future data to set thresholds.
    """

    def _winsorize_row(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        if len(valid) == 0:
            return row
        lo = valid.quantile(lower)
        hi = valid.quantile(upper)
        return row.clip(lower=lo, upper=hi)

    return df.apply(_winsorize_row, axis=1)


# ──────────────────────────────────────────────────
# 3. FAMA-FRENCH FACTORS
# ──────────────────────────────────────────────────

def load_french_factors(
    start: str = data_cfg.start_date,
    end: str = data_cfg.end_date,
) -> pd.DataFrame:
    """
    Download Fama-French 3 factors + risk-free rate + momentum (UMD)
    from the Kenneth French Data Library.

    Returns monthly DataFrame with columns: Mkt-RF, SMB, HML, RF, UMD.
    All values are in decimal (the raw data is in percentage).

    Why UMD matters here: our feature set includes momentum 12-1 and short-term
    reversal, so the long-short book loads on the momentum factor by construction.
    An FF3 regression (which omits momentum) would attribute that loading to alpha.
    The Carhart 4-factor model adds UMD and gives an honest alpha estimate.
    """
    url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Research_Data_Factors_CSV.zip"
    )
    url_mom = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Momentum_Factor_CSV.zip"
    )

    try:
        ff = pd.read_csv(
            url,
            skiprows=3,  # header rows
            index_col=0,
        )
    except Exception as e:
        logger.error(f"Failed to download French data: {e}")
        raise

    # The CSV has both monthly and annual data separated by a blank row.
    # Keep only rows where the index looks like a date (6-digit YYYYMM).
    ff.index = ff.index.astype(str).str.strip()
    ff = ff[ff.index.str.match(r"^\d{6}$")]

    # Parse index to datetime
    ff.index = pd.to_datetime(ff.index, format="%Y%m") + pd.offsets.MonthEnd(0)
    ff.index.name = "date"

    # Convert columns to numeric and from percentage to decimal
    for col in ff.columns:
        ff[col] = pd.to_numeric(ff[col], errors="coerce") / 100.0

    # ── Momentum factor (UMD), distributed as a separate file by French ──
    # The number of preamble lines in this file is not stable across releases,
    # so we probe a few skiprows values rather than hardcoding one, and accept
    # the first parse that yields YYYYMM-looking index entries.
    try:
        mom = None
        for skip in range(10, 18):
            try:
                cand = pd.read_csv(url_mom, skiprows=skip, index_col=0)
            except Exception:
                continue
            idx = cand.index.astype(str).str.strip()
            if idx.str.match(r"^\d{6}$").sum() > 100:  # plausible monthly series
                cand.index = idx
                mom = cand
                break

        if mom is None:
            raise ValueError("could not locate the monthly data block")

        mom = mom[mom.index.str.match(r"^\d{6}$")]
        mom.index = pd.to_datetime(mom.index, format="%Y%m") + pd.offsets.MonthEnd(0)
        mom.index.name = "date"

        mom_col = mom.columns[0]  # single data column, named "Mom" with padding
        ff["UMD"] = pd.to_numeric(mom[mom_col], errors="coerce") / 100.0
        logger.info("Loaded momentum factor (UMD) for Carhart 4-factor regression")
    except Exception as e:
        logger.warning(f"Could not download momentum factor (UMD): {e}. "
                        "Carhart 4-factor regression will be skipped.")

    # Filter date range
    ff = ff.loc[start:end]

    logger.info(f"Loaded French factors: {ff.shape[0]} months, columns: {list(ff.columns)}")
    return ff


# ──────────────────────────────────────────────────
# 4. MARKET CAP (APPROXIMATE)
# ──────────────────────────────────────────────────

def get_market_caps(tickers: list[str]) -> pd.Series:
    """
    Get current market capitalizations from yfinance.

    Limitation: this gives us a STATIC snapshot (today's market cap), not historical.
    For cross-sectional ranking within S&P 100, relative rankings are reasonably stable
    (Apple, Microsoft, etc. have been at the top for years), but this is an approximation.

    In production with CRSP/Compustat, we would use point-in-time market cap (shares
    outstanding × price at each month-end).
    """
    market_caps = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            mcap = info.get("marketCap", np.nan)
            market_caps[ticker] = mcap
        except Exception:
            market_caps[ticker] = np.nan

    result = pd.Series(market_caps, name="market_cap")
    n_valid = result.notna().sum()
    logger.info(f"Retrieved market cap for {n_valid}/{len(tickers)} tickers")
    return result


# ──────────────────────────────────────────────────
# 5. MAIN PIPELINE
# ──────────────────────────────────────────────────

def run_data_pipeline(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute the full data pipeline:
      1. Get S&P 100 tickers
      2. Download prices
      3. Compute and clean returns
      4. Download French factors
      5. Save everything to Parquet

    Returns
    -------
    returns : pd.DataFrame
        Clean monthly returns, shape (n_months, n_tickers)
    prices : pd.DataFrame
        Raw price data (for feature engineering)
    french : pd.DataFrame
        Fama-French factors + RF
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Tickers
    logger.info("=" * 60)
    logger.info("STEP 1: Getting S&P 100 tickers")
    logger.info("=" * 60)
    tickers = get_sp100_tickers()

    # Step 2: Download prices
    logger.info("=" * 60)
    logger.info("STEP 2: Downloading price data")
    logger.info("=" * 60)
    prices = download_price_data(tickers)

    # Step 3: Compute returns
    logger.info("=" * 60)
    logger.info("STEP 3: Computing and cleaning returns")
    logger.info("=" * 60)
    returns = compute_returns(prices)
    returns = clean_returns(returns)
    returns = winsorize_cross_section(returns)
    logger.info("Applied cross-sectional winsorization at 1%/99% per month")

    # Step 4: French factors
    logger.info("=" * 60)
    logger.info("STEP 4: Loading Fama-French factors")
    logger.info("=" * 60)
    french = load_french_factors()

# Align dates: normalize both to month-end
    # yfinance uses varying day-of-month, French uses month-end.
    returns.index = returns.index.to_period("M").to_timestamp("M")
    prices.index = prices.index.to_period("M").to_timestamp("M")
    french.index = french.index.to_period("M").to_timestamp("M")

    common_dates = returns.index.intersection(french.index)
    returns = returns.loc[common_dates]
    french = french.loc[common_dates]
    logger.info(f"Aligned on {len(common_dates)} common dates")

    # Step 5: Save
    logger.info("=" * 60)
    logger.info("STEP 5: Saving to Parquet")
    logger.info("=" * 60)
    prices.to_parquet(data_path / "prices.parquet")
    returns.to_parquet(data_path / "returns.parquet")
    french.to_parquet(data_path / "french_factors.parquet")

    # Save ticker list for reference
    pd.Series(tickers, name="ticker").to_csv(data_path / "tickers.csv", index=False)

    logger.info("Data pipeline complete!")
    logger.info(f"  Returns shape: {returns.shape}")
    logger.info(f"  Date range: {returns.index[0]} to {returns.index[-1]}")
    logger.info(f"  Tickers: {returns.shape[1]}")

    return returns, prices, french


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    returns, prices, french = run_data_pipeline()
