"""
Evaluation
==========
Computes all performance metrics:
  1. Information Coefficient (IC) — rank correlation between predictions and realized returns
  2. Long-short portfolio returns — buy top decile, sell bottom decile
  3. Sharpe ratio — risk-adjusted return of the long-short strategy
  4. Factor-adjusted alpha — does ML add value beyond Fama-French?
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from config import eval_cfg

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────
# 1. INFORMATION COEFFICIENT
# ──────────────────────────────────────────────────

def compute_ic(
    predictions: pd.DataFrame,
    model_names: list[str],
) -> pd.DataFrame:
    """
    Compute monthly Information Coefficient (Spearman rank correlation)
    between predicted and realized returns.

    Parameters
    ----------
    predictions : pd.DataFrame
        Must have columns: date, realized, and one column per model.
    model_names : list of str
        Model column names.

    Returns
    -------
    ic_df : pd.DataFrame
        Monthly IC for each model. Columns: model names. Index: dates.
    """
    ic_records = []

    for date, group in predictions.groupby("date"):
        if len(group) < 10:
            continue  # too few stocks for meaningful correlation

        row = {"date": date}
        for model_name in model_names:
            # Spearman rank correlation
            corr, pval = stats.spearmanr(group[model_name], group["realized"])
            row[model_name] = corr
            row[f"{model_name}_pval"] = pval

        ic_records.append(row)

    ic_df = pd.DataFrame(ic_records).set_index("date")

    return ic_df


def ic_summary(ic_df: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    """
    Summary statistics for ICs.

    Key metrics:
      - Mean IC: average predictive power. IC > 0.03 is considered decent.
      - IC IR (Information Ratio): Mean IC / Std IC. Measures consistency.
        IR > 0.5 is good. An IR of 1.0 means the model is right every month.
      - Hit rate: fraction of months with IC > 0. Above 50% means the model
        is right more often than wrong.
      - t-stat: tests if mean IC is significantly different from zero.
        |t| > 2 → significant at 5% level.
    """
    records = []

    for model_name in model_names:
        ic_series = ic_df[model_name].dropna()

        mean_ic = ic_series.mean()
        std_ic = ic_series.std()
        ic_ir = mean_ic / std_ic if std_ic > 0 else 0
        hit_rate = (ic_series > 0).mean()
        t_stat = mean_ic / (std_ic / np.sqrt(len(ic_series))) if std_ic > 0 else 0

        records.append({
            "model": model_name,
            "mean_IC": mean_ic,
            "std_IC": std_ic,
            "IC_IR": ic_ir,
            "hit_rate": hit_rate,
            "t_stat": t_stat,
            "n_months": len(ic_series),
        })

    return pd.DataFrame(records).set_index("model")


# ──────────────────────────────────────────────────
# 2. LONG-SHORT PORTFOLIO
# ──────────────────────────────────────────────────

def build_long_short_portfolio(
    predictions: pd.DataFrame,
    model_names: list[str],
    n_quantiles: int = eval_cfg.n_quantiles,
    long_q: int = eval_cfg.long_quantile,
    short_q: int = eval_cfg.short_quantile,
) -> pd.DataFrame:
    """
    Build monthly long-short portfolio returns.

    At each date t:
      1. Rank stocks by model predictions
      2. Assign to deciles (1 = lowest predicted return, 10 = highest)
      3. Long portfolio: equal-weight average return of top decile
      4. Short portfolio: equal-weight average return of bottom decile
      5. Long-short return = long - short

    Predictions are formed at the end of month t and earn the return of month t+1,
    so the output is indexed by the month in which the return is earned (t+1).
    Factor regressions, sub-periods and worst months all rely on that label.
    """
    portfolio_returns = []

    for date, group in predictions.groupby("date"):
        if len(group) < n_quantiles:
            continue

        row = {"date": date}

        for model_name in model_names:
            group = group.copy()
            
            # Use rank-based assignment instead of qcut to handle ties robustly
            group["quantile"] = pd.qcut(
                group[model_name].rank(method="first"),
                q=n_quantiles,
                labels=range(1, n_quantiles + 1),
            )

            # Long: top quantile, Short: bottom quantile
            long_mask = group["quantile"] == long_q
            short_mask = group["quantile"] == short_q
            
            if long_mask.any() and short_mask.any():
                long_ret = group.loc[long_mask, "realized"].mean()
                short_ret = group.loc[short_mask, "realized"].mean()
                ls_ret = long_ret - short_ret
            else:
                long_ret = short_ret = ls_ret = np.nan

            row[f"{model_name}_long"] = long_ret
            row[f"{model_name}_short"] = short_ret
            row[f"{model_name}_long_short"] = ls_ret

        portfolio_returns.append(row)

    out = pd.DataFrame(portfolio_returns).set_index("date")
    out.index = out.index + pd.offsets.MonthEnd(1)
    out.index.name = "date"
    return out


def portfolio_summary(
    portfolio_returns: pd.DataFrame,
    model_names: list[str],
    annualization: int = eval_cfg.annualization_factor,
) -> pd.DataFrame:
    """
    Summary statistics for long-short portfolio.

    Metrics:
      - Annualized return: mean monthly × 12
      - Annualized volatility: std monthly × sqrt(12)
      - Sharpe ratio: annualized return / annualized vol
        Note: no RF subtraction for a zero-investment (long-short) portfolio.
        The portfolio is self-financing — you're long $1 and short $1.
      - Max drawdown: largest peak-to-trough decline in cumulative returns
      - Skewness: negative skew means fat left tail (crash risk)
    """
    records = []

    for model_name in model_names:
        col = f"{model_name}_long_short"
        if col not in portfolio_returns.columns:
            continue

        rets = portfolio_returns[col].dropna()

        mean_ret = rets.mean() * annualization
        vol = rets.std() * np.sqrt(annualization)
        sharpe = mean_ret / vol if vol > 0 else 0

        # Cumulative returns for drawdown
        cum = (1 + rets).cumprod()
        peak = cum.cummax()
        drawdown = (cum - peak) / peak
        max_dd = drawdown.min()

        skew = rets.skew()

        records.append({
            "model": model_name,
            "ann_return": mean_ret,
            "ann_vol": vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "skewness": skew,
            "n_months": len(rets),
        })

    return pd.DataFrame(records).set_index("model")


# ──────────────────────────────────────────────────
# 3. FACTOR REGRESSION (FF3 ALPHA)
# ──────────────────────────────────────────────────

def _regress_with_hac(y: np.ndarray, X: np.ndarray, factor_names: list[str],
                       maxlags: int = 6) -> dict:
    """
    OLS with Newey-West (HAC) standard errors.

    Why HAC: monthly long-short factor returns are typically autocorrelated and
    heteroskedastic. Plain OLS standard errors are then downward-biased, which
    inflates the alpha t-statistic. Newey-West corrects for serial correlation
    up to `maxlags`. We use 6 lags, a common choice for monthly data (roughly
    T^(1/4) for T=120), consistent with the BAB replication in this portfolio.

    Returns a dict with the annualized alpha, its HAC t-stat, factor betas,
    the OLS t-stat (for comparison), and R-squared.
    """
    import statsmodels.api as sm

    Xc = sm.add_constant(X, has_constant="add")

    ols = sm.OLS(y, Xc).fit()
    hac = ols.get_robustcov_results(cov_type="HAC", maxlags=maxlags, use_correction=True)

    alpha_monthly = float(hac.params[0])

    out = {
        "alpha_monthly": alpha_monthly,
        "alpha_annual": alpha_monthly * 12,
        "t_stat_alpha_HAC": float(hac.tvalues[0]),
        "t_stat_alpha_OLS": float(ols.tvalues[0]),
        "r_squared": float(ols.rsquared),
        "n_months": int(len(y)),
    }
    for i, fname in enumerate(factor_names):
        out[f"beta_{fname}"] = float(hac.params[i + 1])
        out[f"t_{fname}"] = float(hac.tvalues[i + 1])

    return out


def compute_factor_alpha(
    portfolio_returns: pd.DataFrame,
    french_factors: pd.DataFrame,
    model_names: list[str],
    maxlags: int = 6,
) -> pd.DataFrame:
    """
    Regress long-short returns on factor models to test whether the ML strategy
    generates alpha beyond known risk factors.

    Two specifications are estimated for every model:

      FF3     : R_LS = a + b1*MktRF + b2*SMB + b3*HML + e
      Carhart : R_LS = a + b1*MktRF + b2*SMB + b3*HML + b4*UMD + e

    Why Carhart matters here: the feature set contains momentum 12-1 and
    short-term reversal, so the book loads on the momentum factor by
    construction. FF3 omits UMD and therefore misattributes that loading to
    alpha. The Carhart alpha is the honest number; expect it to be materially
    lower than the FF3 alpha.

    All standard errors are Newey-West (HAC, `maxlags` lags). The OLS t-stat is
    also reported so the size of the correction is visible.

    Note: the long-short book is self-financing, so no risk-free subtraction.
    """
    records = []

    ff3_cols = ["Mkt-RF", "SMB", "HML"]
    carhart_cols = ff3_cols + ["UMD"]
    has_umd = "UMD" in french_factors.columns

    if not has_umd:
        logger.warning("UMD not available - reporting FF3 only (Carhart skipped)")

    for model_name in model_names:
        col = f"{model_name}_long_short"
        if col not in portfolio_returns.columns:
            continue

        common = portfolio_returns.index.intersection(french_factors.index)
        y = portfolio_returns.loc[common, col].dropna()
        if len(y) < 24:
            continue
        dates = y.index

        # ── FF3 ──
        X3 = french_factors.loc[dates, ff3_cols].values
        res3 = _regress_with_hac(y.values, X3, ff3_cols, maxlags=maxlags)

        row = {
            "model": model_name,
            "FF3_alpha_annual": res3["alpha_annual"],
            "FF3_t_HAC": res3["t_stat_alpha_HAC"],
            "FF3_t_OLS": res3["t_stat_alpha_OLS"],
            "FF3_r2": res3["r_squared"],
        }

        # ── Carhart 4-factor ──
        if has_umd:
            sub = french_factors.loc[dates, carhart_cols].dropna()
            if len(sub) >= 24:
                y4 = y.loc[sub.index].values
                res4 = _regress_with_hac(y4, sub.values, carhart_cols, maxlags=maxlags)
                row.update({
                    "Carhart_alpha_annual": res4["alpha_annual"],
                    "Carhart_t_HAC": res4["t_stat_alpha_HAC"],
                    "Carhart_t_OLS": res4["t_stat_alpha_OLS"],
                    "beta_UMD": res4["beta_UMD"],
                    "t_UMD": res4["t_UMD"],
                    "Carhart_r2": res4["r_squared"],
                    "alpha_explained_by_UMD": (
                        res3["alpha_annual"] - res4["alpha_annual"]
                    ),
                })

        row.update({
            "beta_MktRF": res3["beta_Mkt-RF"],
            "beta_SMB": res3["beta_SMB"],
            "beta_HML": res3["beta_HML"],
            "n_months": res3["n_months"],
        })
        records.append(row)

    return pd.DataFrame(records).set_index("model")


# ──────────────────────────────────────────────────
# 4. MASTER EVALUATION
# ──────────────────────────────────────────────────

def evaluate_all(
    predictions: pd.DataFrame,
    french_factors: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Run all evaluations and return a dict of result DataFrames.
    """
    model_names = [
        col for col in predictions.columns
        if col not in ("date", "ticker", "realized")
    ]

    logger.info(f"Evaluating models: {model_names}")

    # IC
    ic_df = compute_ic(predictions, model_names)
    ic_stats = ic_summary(ic_df, model_names)
    logger.info(f"\nIC Summary:\n{ic_stats}")

    # Portfolio
    port_rets = build_long_short_portfolio(predictions, model_names)
    port_stats = portfolio_summary(port_rets, model_names)
    logger.info(f"\nPortfolio Summary:\n{port_stats}")

    results = {
        "ic_monthly": ic_df,
        "ic_summary": ic_stats,
        "portfolio_returns": port_rets,
        "portfolio_summary": port_stats,
    }

    # Factor alpha (if French factors available)
    if french_factors is not None:
        alpha_df = compute_factor_alpha(port_rets, french_factors, model_names)
        logger.info(f"\nFactor Alpha:\n{alpha_df}")
        results["factor_alpha"] = alpha_df

    return results
