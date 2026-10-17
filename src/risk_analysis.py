"""
Risk & Robustness Analysis
==========================
Diagnostics that a backtest headline number hides:

  1. Out-of-sample R-squared  (near zero, sometimes negative - and that is fine)
  2. Sharpe standard errors   (is the Ridge-Lasso gap even distinguishable?)
  3. Sub-period stability     (is the edge concentrated in one regime?)
  4. Worst months + drawdown trough (when did the book actually lose?)
  5. Long/short leg split     (where does the alpha come from?)

Each one answers a question a headline Sharpe cannot.
"""

import logging

import numpy as np
import pandas as pd

from config import eval_cfg

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────
# 1. OUT-OF-SAMPLE R-SQUARED
# ──────────────────────────────────────────────────

def oos_r_squared(predictions: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    """
    Pooled out-of-sample R-squared, benchmarked against the mean realized return.

        R2 = 1 - SS_res / SS_tot,   SS_tot = sum (y - mean(y))^2

    Expect values near zero, and negative for weak models - a negative R2 simply
    means the model predicts worse than a constant. This is the normal state of
    affairs in monthly equity prediction and does NOT contradict a positive IC:
    R-squared scores the level, IC scores the ranking, and only the ranking is
    monetized by a long-short book.

    Note on convention: Gu-Kelly-Xiu (2020) benchmark against a zero prediction
    rather than the sample mean, which is more conservative still. We report the
    mean-benchmarked version and flag the distinction.
    """
    y = predictions["realized"].values
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_tot_zero = float(np.sum(y ** 2))

    rows = []
    for m in model_names:
        ss_res = float(np.sum((y - predictions[m].values) ** 2))
        rows.append({
            "model": m,
            "oos_r2": 1 - ss_res / ss_tot,
            "oos_r2_pct": 100 * (1 - ss_res / ss_tot),
            "oos_r2_vs_zero_pct": 100 * (1 - ss_res / ss_tot_zero),
        })

    return pd.DataFrame(rows).set_index("model")


# ──────────────────────────────────────────────────
# 2. SHARPE RATIO STANDARD ERRORS
# ──────────────────────────────────────────────────

def sharpe_with_ci(returns: pd.Series, annualization: int = 12) -> dict:
    """
    Annualized Sharpe with an asymptotic standard error and 95% CI.

    Lo (2002): for iid returns, SE(SR) ~= sqrt((1 + SR^2 / 2) / T), where SR and
    T are on the SAME frequency. We compute the SE on monthly SR, then annualize
    both the estimate and its SE by sqrt(annualization).

    Why this matters: with T = 120 months, SE on the annualized Sharpe is roughly
    0.1. A gap of 0.1 between two models is therefore inside sampling error and
    must not be reported as one model "beating" another.
    """
    r = returns.dropna()
    T = len(r)
    if T < 12 or r.std() == 0:
        return {"sharpe": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_months": T}

    sr_m = r.mean() / r.std()                      # monthly Sharpe
    se_m = np.sqrt((1 + 0.5 * sr_m ** 2) / T)      # Lo (2002)

    k = np.sqrt(annualization)
    sr_a, se_a = sr_m * k, se_m * k

    return {
        "sharpe": float(sr_a),
        "se": float(se_a),
        "ci_low": float(sr_a - 1.96 * se_a),
        "ci_high": float(sr_a + 1.96 * se_a),
        "n_months": T,
    }


def sharpe_table(portfolio_returns: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    rows = []
    for m in model_names:
        col = f"{m}_long_short"
        if col not in portfolio_returns.columns:
            continue
        rows.append({"model": m, **sharpe_with_ci(portfolio_returns[col])})
    return pd.DataFrame(rows).set_index("model")


# ──────────────────────────────────────────────────
# 3. SUB-PERIOD STABILITY
# ──────────────────────────────────────────────────

def subperiod_analysis(
    portfolio_returns: pd.DataFrame,
    model_names: list[str],
    split_year: int = 2020,
) -> pd.DataFrame:
    """
    Split the out-of-sample window in two and report Sharpe on each half.

    If the edge lives entirely in one half, the full-sample Sharpe is a blend of
    a good regime and a flat one, and the strategy has a regime dependence that
    should be stated up front.
    """
    rows = []
    for m in model_names:
        col = f"{m}_long_short"
        if col not in portfolio_returns.columns:
            continue
        r = portfolio_returns[col].dropna()
        early = r[r.index.year < split_year]
        late = r[r.index.year >= split_year]

        rows.append({
            "model": m,
            f"sharpe_pre{split_year}": sharpe_with_ci(early)["sharpe"],
            f"n_months_pre{split_year}": len(early),
            f"sharpe_post{split_year}": sharpe_with_ci(late)["sharpe"],
            f"n_months_post{split_year}": len(late),
            "sharpe_full": sharpe_with_ci(r)["sharpe"],
        })

    return pd.DataFrame(rows).set_index("model")


# ──────────────────────────────────────────────────
# 4. LEGS, WORST MONTHS, DRAWDOWN TROUGH
# ──────────────────────────────────────────────────

def build_legs(
    predictions: pd.DataFrame,
    model_name: str,
    n_quantiles: int = eval_cfg.n_quantiles,
) -> pd.DataFrame:
    """Monthly long-leg, short-leg and long-short returns for one model, indexed by return month (t+1)."""
    rows = []
    for d, g in predictions.groupby("date"):
        if len(g) < n_quantiles:
            continue
        g = g.copy()
        g["q"] = pd.qcut(
            g[model_name].rank(method="first"),
            q=n_quantiles,
            labels=range(1, n_quantiles + 1),
        )
        lo = g.loc[g["q"] == n_quantiles, "realized"].mean()
        sh = g.loc[g["q"] == 1, "realized"].mean()
        rows.append({"date": d, "long": lo, "short": sh, "long_short": lo - sh})

    out = pd.DataFrame(rows).set_index("date").sort_index()
    out.index = out.index + pd.offsets.MonthEnd(1)
    out.index.name = "date"
    return out


def worst_months(legs: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """The n worst long-short months, with the leg attribution for each."""
    w = legs["long_short"].nsmallest(n)
    return legs.loc[w.index, ["long", "short", "long_short"]]


def drawdown_trough(legs: pd.DataFrame) -> dict:
    """Date and depth of the maximum drawdown trough of the long-short book."""
    cum = (1 + legs["long_short"]).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    return {"trough_date": dd.idxmin(), "max_drawdown": float(dd.min())}


# ──────────────────────────────────────────────────
# 5. LEG-SPLIT FACTOR ALPHA
# ──────────────────────────────────────────────────

def leg_alpha(
    legs: pd.DataFrame,
    french_factors: pd.DataFrame,
    maxlags: int = 6,
) -> pd.DataFrame:
    """
    Regress each leg on the factor model, with Newey-West standard errors.

    Legs are individual asset portfolios, so they are regressed as EXCESS returns
    (minus the risk-free rate). The long-short book is self-financing and is
    regressed as-is.

    Reading the short-leg alpha: a POSITIVE alpha on the short leg means the
    bottom-decile names outperformed their factor exposures, i.e. shorting them
    cost alpha. Its contribution to the long-short alpha is the negative of that
    number, because the book is short those names.

    Why this matters: if the alpha were concentrated on the short leg, the true
    net-of-borrow economics would be worse than a flat cost assumption implies,
    since shorting carries borrow fees, recall risk and margin. Volunteering that
    conclusion is worth more than the split itself.
    """
    import statsmodels.api as sm

    cols = [c for c in ["Mkt-RF", "SMB", "HML", "UMD"] if c in french_factors.columns]
    idx = legs.index.to_period("M").to_timestamp("M")
    L = legs.copy()
    L.index = idx

    common = L.index.intersection(french_factors.dropna(subset=cols).index)
    if len(common) < 24:
        return pd.DataFrame()

    F = french_factors.loc[common, cols].values
    rf = french_factors.loc[common, "RF"].values

    rows = []
    for leg in ["long", "short", "long_short"]:
        y = L.loc[common, leg].values.copy()
        if leg in ("long", "short"):
            y = y - rf  # individual portfolios: excess of risk-free

        X = sm.add_constant(F, has_constant="add")
        hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags},
                                use_t=True)

        rows.append({
            "leg": leg,
            "alpha_annual_pct": 100 * float(hac.params[0]) * 12,
            "t_stat_HAC": float(hac.tvalues[0]),
            "n_months": len(y),
            "spec": "Carhart" if "UMD" in cols else "FF3",
        })

    df = pd.DataFrame(rows).set_index("leg")

    # Contribution of the short book to the long-short alpha (sign flipped)
    if "short" in df.index:
        df.loc["short", "contribution_to_LS_alpha_pct"] = -df.loc["short", "alpha_annual_pct"]

    return df


# ──────────────────────────────────────────────────
# MASTER
# ──────────────────────────────────────────────────

def run_risk_analysis(
    predictions: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    french_factors: pd.DataFrame,
    headline_model: str = "Lasso",
) -> dict:
    """Run every diagnostic and persist the tables to results/."""
    from pathlib import Path

    model_names = [c for c in predictions.columns
                   if c not in ("date", "ticker", "realized")]

    logger.info("=" * 60)
    logger.info("RISK & ROBUSTNESS ANALYSIS")
    logger.info("=" * 60)

    r2 = oos_r_squared(predictions, model_names)
    logger.info(f"\nOut-of-sample R-squared:\n{r2.round(4).to_string()}")

    sh = sharpe_table(portfolio_returns, model_names)
    logger.info(f"\nSharpe with 95% CI (Lo 2002):\n{sh.round(3).to_string()}")
    logger.info("  Note: gaps smaller than ~2x SE are inside sampling error.")

    sub = subperiod_analysis(portfolio_returns, model_names)
    logger.info(f"\nSub-period Sharpe:\n{sub.round(3).to_string()}")

    legs = build_legs(predictions, headline_model)
    wm = worst_months(legs)
    tr = drawdown_trough(legs)
    logger.info(f"\n{headline_model} - three worst months (long-short):")
    for d, row in wm.iterrows():
        logger.info(f"  {d:%Y-%m}: {100*row['long_short']:+6.2f}%  "
                     f"(long {100*row['long']:+6.2f}% / short {100*row['short']:+6.2f}%)")
    logger.info(f"  Max-drawdown trough: {tr['trough_date']:%Y-%m} "
                 f"({100*tr['max_drawdown']:.1f}%)")

    la = leg_alpha(legs, french_factors)
    if not la.empty:
        logger.info(f"\n{headline_model} - leg-split alpha ({la['spec'].iloc[0]}, HAC):\n"
                     f"{la.round(2).to_string()}")

    Path("results").mkdir(exist_ok=True)
    r2.to_csv("results/oos_r_squared.csv")
    sh.to_csv("results/sharpe_confidence_intervals.csv")
    sub.to_csv("results/subperiod_sharpe.csv")
    wm.to_csv("results/worst_months.csv")
    if not la.empty:
        la.to_csv("results/leg_split_alpha.csv")
    logger.info("\nSaved risk tables to results/")

    return {"oos_r2": r2, "sharpe_ci": sh, "subperiod": sub,
            "worst_months": wm, "trough": tr, "leg_alpha": la, "legs": legs}
