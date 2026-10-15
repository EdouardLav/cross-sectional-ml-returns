"""
Nested Walk-Forward Cross-Validation
====================================
Selects hyperparameters using an INNER walk-forward on the training set only,
nested inside the OUTER walk-forward that produces out-of-sample predictions.

Why nested
----------
Choosing a hyperparameter by looking at test-set performance turns the test set
into a training set: the reported Sharpe becomes the maximum over N noisy draws,
which is optimistic even when no configuration has real edge (Varma & Simon,
2006). Nested CV evaluates the *procedure* (select-then-predict), which is what
would actually be deployed, rather than a single hand-picked configuration.

Why symmetric tuning
--------------------
The headline result of this project is that linear models beat tree models. If
only the linear models were tuned, the obvious objection is "you tuned the
winners". Every model is therefore tuned by the same inner procedure, on the
same folds, with the same selection criterion.

Structure, for outer fold Y (e.g. Y = 2015, train = [2006, 2014]):

    inner walk-forward on [2006, 2014] ONLY:
        train [2006, 2010] -> validate 2011
        train [2006, 2011] -> validate 2012
        train [2006, 2012] -> validate 2013
        train [2006, 2013] -> validate 2014
    score each candidate config by mean validation IC
    pick the best, refit on all of [2006, 2014], predict Y

Year Y never influences the choice. Splits are by calendar year, never mid-month,
so a cross-section is never divided across train and validation.

Selection criterion
-------------------
Mean Information Coefficient (Spearman rank correlation), not MSE. The economic
objective is cross-sectional ranking, not level prediction; sklearn's default
(MSE / R^2) optimizes the wrong quantity here. `score_by="mse"` is available to
verify the choice is not criterion-driven.

Feasibility rule (important)
----------------------------
A configuration that produces (near-)constant predictions cannot rank a
cross-section: decile assignment on a constant vector degenerates to row order,
i.e. an accidental alphabetical portfolio. Such a configuration is OUTSIDE the
strategy's feasible set. It is therefore DISQUALIFIED from selection (scored
-inf), not treated as a legitimate "zero-signal" candidate. Without this rule,
Lasso can select an alpha that zeroes every coefficient, freeze the book on an
arbitrary ticker ordering, and contaminate the out-of-sample statistics.
"""

import logging
from itertools import product
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from config import model_cfg

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────
# HYPERPARAMETER GRIDS
# ──────────────────────────────────────────────────
# Log-spaced for the penalties. Lasso's grid sits lower: L1 zeroes out every
# coefficient aggressively at this signal-to-noise ratio. Configurations that
# do zero everything are handled by the feasibility rule above, not by grid
# surgery, so the grid can stay wide.
#
# Tree grids are deliberately small. Each extra point multiplies the inner-fold
# cost, and with a negative out-of-sample R-squared the binding constraint on
# tree performance is capacity, not fine-grained hyperparameter placement.

PARAM_GRIDS = {
    "Ridge": {
        # Extended upward after the selection pinned at the previous ceiling
        # (100) on every outer fold. As alpha grows, Ridge predictions converge
        # (up to scale) toward the covariance-weighted combination X'y, so the
        # IC plateaus: a selection landing on that plateau is informative
        # ("the inner CV wants heavy shrinkage"), not pathological. Unlike
        # Lasso, Ridge never becomes exactly constant, so no degeneracy risk.
        "alpha": list(np.logspace(-2, 4, 13)),          # 1e-2 ... 1e4
    },
    "Lasso": {
        "alpha": list(np.logspace(-6, -1, 11)),         # 1e-6 ... 1e-1
    },
    "RandomForest": {
        "max_depth": [3, 5, 8],
        "min_samples_leaf": [5, 20, 50],
    },
    "XGBoost": {
        "max_depth": [2, 3, 5],
        "learning_rate": [0.01, 0.05],
        "reg_lambda": [1.0, 10.0],
    },
}

# Minimum years of inner-training data before the first inner validation year.
# With train_start=2006 and 5 years of burn-in, the first outer fold (2015)
# validates on 2011..2014 -> 4 inner folds. Later folds get progressively more.
MIN_INNER_TRAIN_YEARS = 5


def _build(model_name: str, params: dict):
    """Instantiate a model with a candidate configuration, seeded for reproducibility."""
    seed = model_cfg.random_seed

    if model_name == "Ridge":
        return Ridge(**params)

    if model_name == "Lasso":
        return Lasso(max_iter=10000, **params)

    if model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=model_cfg.rf_n_estimators,
            max_features=model_cfg.rf_max_features,
            random_state=seed,
            n_jobs=-1,
            **params,
        )

    if model_name == "XGBoost":
        return XGBRegressor(
            n_estimators=model_cfg.xgb_n_estimators,
            subsample=model_cfg.xgb_subsample,
            colsample_bytree=model_cfg.xgb_colsample_bytree,
            reg_alpha=model_cfg.xgb_reg_alpha,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
            **params,
        )

    raise ValueError(f"Unknown model: {model_name}")


# ──────────────────────────────────────────────────
# SCORING
# ──────────────────────────────────────────────────

def _mean_ic(df: pd.DataFrame, preds: np.ndarray, target_col: str) -> float:
    """
    Mean cross-sectional Spearman IC across the dates in `df`.

    IC is computed per date (one cross-section), then averaged. Pooling all
    dates into a single correlation would conflate cross-sectional ranking with
    time-series variation in the average return level.

    Feasibility rule: a (near-)constant prediction vector carries no ranking
    information, and downstream decile assignment on it degenerates to row
    order. A configuration that is degenerate on a majority of validation
    months is disqualified (-inf) rather than scored as "no signal" - it is
    outside the strategy's feasible set, not a neutral candidate.
    """
    tmp = df[["date", target_col]].copy()
    tmp["pred"] = preds

    ics = []
    n_dates = 0
    n_degenerate = 0
    for _, g in tmp.groupby("date"):
        if len(g) < 10:
            continue
        n_dates += 1
        if g["pred"].nunique() < 3 or g["pred"].std() < 1e-12:
            n_degenerate += 1
            continue
        ic, _ = stats.spearmanr(g["pred"], g[target_col])
        if np.isfinite(ic):
            ics.append(ic)

    if n_dates == 0 or n_degenerate > n_dates / 2:
        return -np.inf

    return float(np.mean(ics)) if ics else -np.inf


def _neg_mse(df: pd.DataFrame, preds: np.ndarray, target_col: str) -> float:
    """Negative MSE, so 'higher is better' holds for both criteria."""
    return -float(np.mean((df[target_col].values - preds) ** 2))


# ──────────────────────────────────────────────────
# INNER WALK-FORWARD
# ──────────────────────────────────────────────────

def select_hyperparams(
    train_panel: pd.DataFrame,
    feature_names: list[str],
    model_name: str,
    target_col: str = "forward_return",
    score_by: Literal["ic", "mse"] = "ic",
    min_inner_train_years: int = MIN_INNER_TRAIN_YEARS,
) -> tuple[dict, pd.DataFrame]:
    """
    Select hyperparameters via an inner walk-forward restricted to `train_panel`.

    Parameters
    ----------
    train_panel : pd.DataFrame
        The OUTER fold's training data ONLY. Must contain a 'year' column.
        The outer test year must not be present.
    feature_names : list of str
    model_name : one of PARAM_GRIDS
    score_by : "ic" (default) or "mse"

    Returns
    -------
    best_params : dict
    scores : pd.DataFrame  (one row per candidate config)
    """
    grid = PARAM_GRIDS[model_name]
    scorer = _mean_ic if score_by == "ic" else _neg_mse

    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*[grid[k] for k in keys])]

    years = sorted(train_panel["year"].unique())
    valid_years = [y for y in years if y >= years[0] + min_inner_train_years]

    if not valid_years:
        fallback = combos[len(combos) // 2]
        logger.warning(f"    {model_name}: too little history for inner CV, "
                        f"defaulting to {fallback}")
        return fallback, pd.DataFrame()

    # Pre-slice the inner folds once, instead of re-filtering per candidate.
    folds = []
    for vy in valid_years:
        tr = train_panel[train_panel["year"] < vy]
        va = train_panel[train_panel["year"] == vy]
        if len(tr) and len(va):
            folds.append((tr[feature_names].values, tr[target_col].values, va))

    records = []
    for params in combos:
        fold_scores = []
        for X_tr, y_tr, va in folds:
            m = _build(model_name, params)
            m.fit(X_tr, y_tr)
            preds = m.predict(va[feature_names].values)
            fold_scores.append(scorer(va, preds, target_col))

        if fold_scores:
            # If ANY inner fold disqualifies the config, the config is out:
            # a strategy that cannot form deciles in some validation year is
            # not deployable, whatever its average score elsewhere.
            if any(s == -np.inf for s in fold_scores):
                mean_score, std_score = -np.inf, np.nan
            else:
                mean_score = float(np.mean(fold_scores))
                std_score = float(np.std(fold_scores))
            records.append({
                **params,
                "mean_score": mean_score,
                "std_score": std_score,
                "n_inner_folds": len(fold_scores),
            })

    scores = pd.DataFrame(records)
    if scores.empty:
        return combos[len(combos) // 2], scores

    feasible = scores[scores["mean_score"] > -np.inf]
    if feasible.empty:
        # Every candidate degenerate (cannot happen for Ridge/trees; for Lasso
        # only if the whole grid zeroes out). Fall back to the LEAST regularized
        # configuration, which is the least likely to be degenerate at refit.
        fallback = combos[0]
        logger.warning(f"    {model_name}: all configs disqualified by the "
                        f"feasibility rule, falling back to {fallback}")
        return fallback, scores

    best_row = feasible.loc[feasible["mean_score"].idxmax()]
    best_params = {k: best_row[k] for k in keys}

    # Restore native types (pandas may upcast ints to float64)
    for k, v in best_params.items():
        if all(isinstance(x, (int, np.integer)) for x in grid[k]):
            best_params[k] = int(v)
        else:
            best_params[k] = float(v)

    n_disq = int((scores["mean_score"] == -np.inf).sum())
    pretty = ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in best_params.items())
    logger.info(
        f"    {model_name}: {pretty} "
        f"({score_by}={best_row['mean_score']:.4f}, "
        f"{int(best_row['n_inner_folds'])} inner folds, "
        f"validated {valid_years[0]}-{valid_years[-1]}"
        + (f", {n_disq} configs disqualified" if n_disq else "")
        + ")"
    )

    return best_params, scores


def build_selected(model_name: str, params: dict):
    """Public helper: build a model from a selected configuration."""
    return _build(model_name, params)


# ──────────────────────────────────────────────────
# REPORTING
# ──────────────────────────────────────────────────

def plot_hyperparam_trajectory(param_history: dict, save: bool = True):
    """
    Plot the selected regularization strength per outer fold, for the linear models.

    Interpretation: a rising alpha over time means the inner walk-forward demands
    more shrinkage as more recent data enters the training window, i.e. the
    signal-to-noise ratio is deteriorating. A flat trajectory means the optimal
    regularization is stable across the sample.

    param_history : {model_name: {test_year: params_dict}}
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    linear = {m: h for m, h in param_history.items()
              if m in ("Ridge", "Lasso") and h}
    if not linear:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"Ridge": "#2c3e50", "Lasso": "#e74c3c"}

    for model_name, per_year in linear.items():
        yrs = sorted(per_year.keys())
        vals = [per_year[y]["alpha"] for y in yrs]
        ax.plot(yrs, vals, marker="o", linewidth=2,
                color=colors[model_name], label=model_name)

    ax.set_yscale("log")
    ax.set_xlabel("Outer test year")
    ax.set_ylabel("Selected alpha (log scale)")
    ax.set_title("Regularization Strength Selected by Inner Walk-Forward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        Path("figures").mkdir(exist_ok=True)
        fig.savefig("figures/alpha_trajectory.png", dpi=150, bbox_inches="tight")
        logger.info("Saved figures/alpha_trajectory.png")

    return fig


def summarize_param_history(param_history: dict) -> pd.DataFrame:
    """Flatten {model: {year: params}} into a tidy DataFrame for results/."""
    rows = []
    for model_name, per_year in param_history.items():
        for year, params in sorted(per_year.items()):
            rows.append({"model": model_name, "test_year": year, **params})
    return pd.DataFrame(rows)