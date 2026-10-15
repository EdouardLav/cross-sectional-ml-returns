"""
Feature Engineering
===================
Computes cross-sectional features from monthly return and volume data.

Timing convention (CRITICAL for avoiding look-ahead bias):
  - At the END of month t, we observe price P_t and compute:
      * return_t = P_t / P_{t-1} - 1  (known at end of month t)
  - Features are computed using data up to and including month t.
  - Target: return_{t+1} = P_{t+1} / P_t - 1  (realized next month)
  - The model predicts return_{t+1} using features known at end of month t.

Cross-sectional standardization:
  At each date t, features are standardized to mean=0, std=1 across all stocks.
  This ensures the model learns relative rankings, not absolute levels.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import feature_cfg, data_cfg

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────
# INDIVIDUAL FEATURE FUNCTIONS
# ──────────────────────────────────────────────────

def compute_momentum_12_1(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum 12-1 (Jegadeesh & Titman 1993):
      Cumulative return from t-12 to t-1, skipping the most recent month.

    At the end of month t:
      mom = (1 + r_{t-11}) * (1 + r_{t-10}) * ... * (1 + r_{t-1}) - 1
          = P_{t-1} / P_{t-12} - 1

    The skip of the last month avoids contamination from short-term reversal.
    """
    lookback = feature_cfg.momentum_lookback
    skip = feature_cfg.momentum_skip

    # Cumulative return over [t-lookback, t-skip)
    # rolling(lookback - skip) on (1+r), then take the product
    cum_ret = (1 + returns).rolling(window=lookback - skip).apply(
        lambda x: x.prod() - 1, raw=True
    )

    # Shift by `skip` months to align: at time t, we use returns up to t-skip
    momentum = cum_ret.shift(skip)

    return momentum


def compute_reversal_1m(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Short-term reversal (Jegadeesh 1990):
      The return of the most recent month, r_t.

    Stocks that went up last month tend to go down this month (and vice versa)
    at the monthly horizon. This is one of the most robust anomalies in finance.

    No shift needed: return at time t is known at the end of month t.
    """
    return returns.copy()


def compute_realized_volatility(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Realized volatility:
      Rolling standard deviation of monthly returns over the past N months.

    Uses monthly returns (not daily) for consistency with our data frequency.
    In production, you'd use daily returns for a more precise estimate.
    """
    window = feature_cfg.volatility_window
    return returns.rolling(window=window, min_periods=window).std()


def compute_max_return(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Maximum monthly return over past N months (Bali, Cakici, Whitelaw 2011).

    Stocks with "lottery-like" characteristics (high MAX) tend to be overpriced
    and subsequently underperform. This captures investor preference for
    positive skewness / speculative demand.
    """
    window = feature_cfg.max_return_window
    return returns.rolling(window=window, min_periods=window).max()


def compute_return_skewness(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling skewness of monthly returns.

    Negative skewness (fat left tail) → crash risk → higher expected returns (risk premium).
    Positive skewness (lottery tickets) → overpricing → lower expected returns.

    Uses pandas .skew() which computes the unbiased sample skewness (Fisher's definition).
    """
    window = feature_cfg.skewness_window
    return returns.rolling(window=window, min_periods=window).skew()


def compute_abnormal_volume(volume: pd.DataFrame) -> pd.DataFrame:
    """
    Abnormal volume: ratio of short-term to long-term average volume.

    High abnormal volume often accompanies information arrival and can
    predict reversals (Campbell, Grossman, Wang 1993).

    Values > 1: recent volume is above the historical average (unusual activity).
    Values < 1: recent volume is below average (quiet period).
    """
    short_w = feature_cfg.volume_short_window
    long_w = feature_cfg.volume_long_window

    vol_short = volume.rolling(window=short_w, min_periods=short_w).mean()
    vol_long = volume.rolling(window=long_w, min_periods=long_w).mean()

    # Avoid division by zero
    abnormal = vol_short / vol_long.replace(0, np.nan)

    return abnormal


def compute_log_market_cap(
    market_caps: pd.Series,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Log market capitalization.

    NOT USED in the pipeline (main.py passes market_caps=None). This static
    feature leaks look-ahead information and was removed after the audit
    described in the README (Audit 1); it is kept only to reproduce that audit.

    Limitation: we use current (static) market caps as an approximation.
    The resulting feature is constant across time — it captures only the
    cross-sectional size ranking, not its evolution.

    In production with CRSP, you'd compute log(shares_outstanding_t * price_t)
    at each month-end.
    """
    log_mcap = np.log(market_caps.replace(0, np.nan))

    # Broadcast to a DataFrame with the same shape as returns (same value each month)
    result = pd.DataFrame(
        np.tile(log_mcap.values, (len(returns.index), 1)),
        index=returns.index,
        columns=returns.columns,
    )

    # Only keep tickers present in both
    common = result.columns.intersection(market_caps.index)
    result = result[common]

    return result


# ──────────────────────────────────────────────────
# CROSS-SECTIONAL STANDARDIZATION
# ──────────────────────────────────────────────────

def standardize_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize each row (cross-section) to mean=0, std=1.

    At each date t:
      z_{i,t} = (x_{i,t} - mean_t) / std_t

    where mean_t and std_t are computed across all stocks i at date t.

    This ensures the model sees relative rankings, not absolute levels.
    Without standardization, features with large magnitudes (e.g., market cap)
    would dominate in linear models.
    """
    row_mean = df.mean(axis=1)
    row_std = df.std(axis=1)

    # Avoid division by zero
    row_std = row_std.replace(0, np.nan)

    standardized = df.sub(row_mean, axis=0).div(row_std, axis=0)

    return standardized


# ──────────────────────────────────────────────────
# MAIN FEATURE PIPELINE
# ──────────────────────────────────────────────────

def build_features(
    returns: pd.DataFrame,
    volume: Optional[pd.DataFrame] = None,
    market_caps: Optional[pd.Series] = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute all features and standardize cross-sectionally.

    Parameters
    ----------
    returns : pd.DataFrame
        Monthly returns, shape (n_months, n_tickers).
    volume : pd.DataFrame, optional
        Monthly volume data, shape (n_months, n_tickers).
    market_caps : pd.Series, optional
        Current market caps per ticker.

    Returns
    -------
    features : dict[str, pd.DataFrame]
        Keys are feature names, values are DataFrames of standardized features.
        Each DataFrame has shape (n_months, n_tickers).
    """
    logger.info("Computing features...")

    raw_features = {}

    # Price-based features (always available)
    raw_features["momentum_12_1"] = compute_momentum_12_1(returns)
    raw_features["reversal_1m"] = compute_reversal_1m(returns)
    raw_features["realized_vol"] = compute_realized_volatility(returns)
    raw_features["max_return"] = compute_max_return(returns)
    raw_features["return_skewness"] = compute_return_skewness(returns)

    # Volume-based features (optional)
    if volume is not None:
        raw_features["abnormal_volume"] = compute_abnormal_volume(volume)
    else:
        logger.warning("No volume data — skipping abnormal_volume feature")

    # Size (optional, static approximation)
    if market_caps is not None:
        raw_features["log_market_cap"] = compute_log_market_cap(market_caps, returns)
    else:
        logger.warning("No market cap data — skipping log_market_cap feature")

    # Cross-sectional standardization
    logger.info("Standardizing features cross-sectionally...")
    features = {}
    for name, feat_df in raw_features.items():
        features[name] = standardize_cross_section(feat_df)
        n_valid = features[name].notna().sum().sum()
        n_total = features[name].size
        logger.info(f"  {name}: {n_valid}/{n_total} valid observations "
                     f"({100 * n_valid / n_total:.1f}%)")

    return features


def features_to_panel(
    features: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert features dict + forward returns into a single panel DataFrame
    suitable for model training.

    The panel has columns: date, ticker, feature_1, ..., feature_n, forward_return.

    Forward returns: at each date t, the forward return is r_{t+1}.
    This is the TARGET variable — what we're predicting.

    CRITICAL: forward_return is computed by shifting returns by -1.
    At row with date t, forward_return = returns at t+1. No look-ahead bias
    because we shift the realized returns backward in time alignment.
    """
    # Compute forward returns (shift returns by -1 so that at date t, we have r_{t+1})
    fwd_ret = forward_returns.shift(-1)

    # Stack all features into a long-form panel
    panels = []

    for name, feat_df in features.items():
        stacked = feat_df.stack()
        stacked.name = name
        panels.append(stacked)

    # Add forward return
    fwd_stacked = fwd_ret.stack()
    fwd_stacked.name = "forward_return"
    panels.append(fwd_stacked)

    # Merge all into one DataFrame
    panel = pd.concat(panels, axis=1)
    panel.index.names = ["date", "ticker"]
    panel = panel.reset_index()

    # Drop rows with any NaN (conservative: ensures all features are available)
    n_before = len(panel)
    panel = panel.dropna()
    n_after = len(panel)
    logger.info(
        f"Panel: {n_after} observations ({n_before - n_after} dropped due to NaN), "
        f"{len(features)} features, "
        f"dates: {panel['date'].min()} to {panel['date'].max()}"
    )

    return panel


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    # Quick test with saved data
    returns = pd.read_parquet("data/returns.parquet")
    features = build_features(returns)
    panel = features_to_panel(features, returns)
    print(panel.head())
    print(f"\nPanel shape: {panel.shape}")
    print(f"Date range: {panel['date'].min()} to {panel['date'].max()}")
