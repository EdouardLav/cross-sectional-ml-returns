"""
Central configuration for Cross-Sectional ML Return Prediction.

All parameters are gathered here so that:
1. Every choice is documented in one place
2. Sensitivity analysis is easy (change one value, rerun)
3. No magic numbers scattered across the codebase
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Data acquisition and cleaning parameters."""

     #--- Universe ---
    # S&P 100: largest, most liquid US stocks. ~100 tickers gives 10 per decile.
    # Trade-off: fewer stocks = noisier decile sorts, but cleaner data & faster iteration.
    # Survivorship bias: we use the CURRENT S&P 100 list, so stocks that were in the index
    # historically but dropped out (bankruptcy, acquisition) are excluded. This biases
    # average returns upward and understates tail risk. With CRSP/Compustat, we would use
    # point-in-time constituents.
    universe: str = "SP100"

    # --- Time period ---
    # 2005-2025 gives ~20 years. First 12 months lost to feature computation (rolling windows).
    # Effective data: 2006-01 to 2025-12 for features, with walk-forward starting 2015.
    start_date: str = "2005-01-01"
    end_date: str = "2025-12-31"

    # --- Cleaning ---
    # Minimum months of data required per ticker. Tickers with fewer observations are dropped.
    # 60 months = 5 years minimum. Ensures enough history for rolling features.
    min_months_per_ticker: int = 60

    # Minimum tickers per month. Months with fewer stocks are dropped.
    # 50 tickers minimum ensures decile sorts have at least 5 stocks each.
    min_tickers_per_month: int = 50

    # Winsorization percentiles (cross-sectional, applied per month).
    # 1%/99% is the academic standard (Fama-French, Gu-Kelly-Xiu 2020).
    # Trade-off: winsorize too aggressively = lose real signal in tails;
    # too little = outliers dominate model fitting.
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99

    # Maximum absolute monthly return. Anything beyond this is likely a data error.
    max_abs_return: float = 2.0  # 200%


@dataclass
class FeatureConfig:
    """Feature engineering parameters."""

    # --- Momentum 12-1 (Jegadeesh & Titman 1993) ---
    # Cumulative return from t-12 to t-1, skipping the most recent month.
    # The skip avoids the short-term reversal effect contaminating momentum.
    momentum_lookback: int = 12  # months
    momentum_skip: int = 1  # months to skip (most recent)

    # --- Short-term reversal ---
    # 1-month lagged return. Captures the well-documented mean-reversion at
    # the monthly horizon (Jegadeesh 1990).
    # This is simply the return at t-1.

    # --- Realized volatility ---
    # Rolling standard deviation of monthly returns.
    volatility_window: int = 12  # months

    # --- Maximum return in past N months (Bali, Cakici, Whitelaw 2011) ---
    # The maximum monthly return over the past 12 months. Stocks with lottery-like
    # payoffs (high MAX) tend to underperform. Captures investor preference for skewness.
    max_return_window: int = 12

    # --- Return skewness ---
    # Rolling skewness of monthly returns. Related to the MAX effect.
    skewness_window: int = 12

    # --- Volume features ---
    # Abnormal volume: ratio of recent average volume to longer-term average.
    # High abnormal volume often precedes reversals (Campbell, Grossman, Wang 1993).
    volume_short_window: int = 3  # months
    volume_long_window: int = 12  # months

    # --- Feature list (for iteration and logging) ---
    feature_names: List[str] = field(default_factory=lambda: [
        "momentum_12_1",
        "reversal_1m",
        "realized_vol",
        "max_return",
        "return_skewness",
        "abnormal_volume",
        # "log_market_cap" removed: static snapshot of current market cap
        # creates look-ahead bias (today's cap embeds test-period performance)
    ])


@dataclass
class ModelConfig:
    """Model hyperparameters."""

    # --- Random seed for reproducibility ---
    random_seed: int = 42

    # --- Ridge Regression ---
    # L2 regularization. Alpha controls regularization strength.
    ridge_alpha: float = 1.0

    # --- Lasso Regression ---
    # L1 regularization. Performs feature selection (some coefficients go to zero).
    lasso_alpha: float = 0.0001

    # --- Random Forest ---
    rf_n_estimators: int = 200
    rf_max_depth: int = 5
    rf_max_features: str = "sqrt"
    rf_min_samples_leaf: int = 5

    # --- XGBoost ---
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_alpha: float = 0.1  # L1 on weights
    xgb_reg_lambda: float = 1.0  # L2 on weights


@dataclass
class WalkForwardConfig:
    """Walk-forward cross-validation parameters."""

    # --- Expanding window ---
    # Train on all data from start to year Y-1, predict all months of year Y.
    train_start_year: int = 2006  # first year with complete features (after 12-month lookback)
    test_start_year: int = 2015  # first year predicted out-of-sample
    test_end_year: int = 2024  # last full year of predictions

    # Minimum training years before first prediction.
    # 2006-2014 = 9 years of training. Enough for tree models to learn patterns.
    min_train_years: int = 9


@dataclass
class EvaluationConfig:
    """Portfolio construction and evaluation parameters."""

    # --- Decile portfolios ---
    # Long top decile (10%), short bottom decile (10%).
    # With ~100 stocks, each decile has ~10 stocks. This is thin but standard.
    n_quantiles: int = 10
    long_quantile: int = 10  # top decile (highest predicted return)
    short_quantile: int = 1  # bottom decile (lowest predicted return)

    # Annualization factor (monthly -> annual)
    annualization_factor: int = 12


# --- Global config instance ---
data_cfg = DataConfig()
feature_cfg = FeatureConfig()
model_cfg = ModelConfig()
wf_cfg = WalkForwardConfig()
eval_cfg = EvaluationConfig()
