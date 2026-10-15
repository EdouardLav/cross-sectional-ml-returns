"""
Utilities
=========
Plotting functions and helpers for results visualization.

All figures are saved to the figures/ directory for the README and notebooks.
Style: clean, minimal, publication-quality. No chartjunk.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logger = logging.getLogger(__name__)

# ── Plot style ──
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#2c3e50", "#e74c3c", "#27ae60", "#f39c12"]  # dark blue, red, green, orange
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


def plot_ic_timeseries(ic_df: pd.DataFrame, model_names: list[str], save: bool = True):
    """
    Plot monthly IC over time with rolling 12-month average.
    Shows whether predictive power is stable or episodic.
    """
    n_models = len(model_names)
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 3 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]

    for ax, model_name, color in zip(axes, model_names, COLORS):
        ic = ic_df[model_name]

        # Monthly IC as bars
        positive = ic >= 0
        ax.bar(ic.index[positive], ic[positive], width=25, color=color, alpha=0.4, label="Monthly IC")
        ax.bar(ic.index[~positive], ic[~positive], width=25, color="grey", alpha=0.4)

        # Rolling 12-month average
        rolling_ic = ic.rolling(12, min_periods=6).mean()
        ax.plot(rolling_ic.index, rolling_ic, color=color, linewidth=2, label="12M rolling avg")

        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel("IC (Spearman)")
        ax.set_title(f"{model_name} — Mean IC: {ic.mean():.4f}, IR: {ic.mean() / ic.std():.2f}")
        ax.legend(loc="upper right", fontsize=8)

    plt.xlabel("Date")
    plt.tight_layout()

    if save:
        fig.savefig(FIG_DIR / "ic_timeseries.png", dpi=150, bbox_inches="tight")
        logger.info("Saved figures/ic_timeseries.png")

    return fig


def plot_cumulative_returns(
    portfolio_returns: pd.DataFrame,
    model_names: list[str],
    save: bool = True,
):
    """
    Plot cumulative returns of long-short portfolios.
    This is the money plot — shows the economic value of predictions.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for model_name, color in zip(model_names, COLORS):
        col = f"{model_name}_long_short"
        if col not in portfolio_returns.columns:
            continue

        cum_ret = (1 + portfolio_returns[col]).cumprod()
        ax.plot(cum_ret.index, cum_ret, color=color, linewidth=2, label=model_name)

    ax.axhline(y=1, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Cumulative Return ($1 invested)")
    ax.set_xlabel("Date")
    ax.set_title("Long-Short Portfolio Cumulative Returns (Top vs Bottom Decile)")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))

    plt.tight_layout()

    if save:
        fig.savefig(FIG_DIR / "cumulative_returns.png", dpi=150, bbox_inches="tight")
        logger.info("Saved figures/cumulative_returns.png")

    return fig


def plot_feature_importance(
    feature_importance: dict,
    model_names: list[str],
    save: bool = True,
):
    """
    Plot average feature importance across all walk-forward folds.
    Shows which features the models rely on most.
    """
    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 5))
    if len(model_names) == 1:
        axes = [axes]

    for ax, model_name, color in zip(axes, model_names, COLORS):
        fi_dict = feature_importance.get(model_name, {})
        if not fi_dict:
            continue

        # Average importance across years
        fi_df = pd.DataFrame(fi_dict).T  # rows=years, cols=features
        avg_fi = fi_df.mean().sort_values(ascending=True)

        ax.barh(avg_fi.index, avg_fi.values, color=color, alpha=0.7)
        ax.set_title(f"{model_name}")
        ax.set_xlabel("Average Importance")

    plt.suptitle("Feature Importance (Average Across Walk-Forward Folds)", fontsize=14)
    plt.tight_layout()

    if save:
        fig.savefig(FIG_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
        logger.info("Saved figures/feature_importance.png")

    return fig


def plot_ic_comparison_bar(ic_summary_df: pd.DataFrame, save: bool = True):
    """
    Bar chart comparing mean IC, IC IR, and hit rate across models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    metrics = [("mean_IC", "Mean IC"), ("IC_IR", "IC Information Ratio"), ("hit_rate", "Hit Rate")]

    for ax, (col, title) in zip(axes, metrics):
        values = ic_summary_df[col]
        bars = ax.bar(values.index, values.values, color=COLORS[: len(values)], alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel(title)

        # Add value labels on bars
        for bar, val in zip(bars, values.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    if "hit_rate" in ic_summary_df.columns:
        axes[2].axhline(y=0.5, color="red", linestyle="--", linewidth=1, label="Random (50%)")
        axes[2].legend()

    plt.suptitle("Model Comparison — Information Coefficient Metrics", fontsize=14)
    plt.tight_layout()

    if save:
        fig.savefig(FIG_DIR / "ic_comparison.png", dpi=150, bbox_inches="tight")
        logger.info("Saved figures/ic_comparison.png")

    return fig


def plot_decile_returns(
    predictions: pd.DataFrame,
    model_name: str,
    n_quantiles: int = 10,
    save: bool = True,
):
    """
    Bar chart of average realized returns by predicted decile.

    If the model works, we should see a monotonic increase from decile 1 to 10:
    stocks predicted to perform worst (decile 1) actually underperform, and stocks
    predicted to perform best (decile 10) actually outperform.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    decile_returns = []
    for date, group in predictions.groupby("date"):
        if len(group) < n_quantiles:
            continue
        group = group.copy()
        group["decile"] = pd.qcut(
            group[model_name].rank(method="first"),
            q=n_quantiles,
            labels=range(1, n_quantiles + 1),
        )
        for d in range(1, n_quantiles + 1):
            mask = group["decile"] == d
            if mask.any():
                decile_returns.append({"decile": d, "return": group.loc[mask, "realized"].mean()})

    dr_df = pd.DataFrame(decile_returns)
    avg_by_decile = dr_df.groupby("decile")["return"].mean() * 12 * 100  # annualized, in %

    colors_decile = ["#e74c3c" if d <= 3 else "#27ae60" if d >= 8 else "#bdc3c7"
                     for d in avg_by_decile.index]
    ax.bar(avg_by_decile.index, avg_by_decile.values, color=colors_decile, alpha=0.8)
    ax.set_xlabel("Predicted Return Decile")
    ax.set_ylabel("Average Annualized Return (%)")
    ax.set_title(f"{model_name} — Realized Returns by Predicted Decile")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(range(1, n_quantiles + 1))

    plt.tight_layout()

    if save:
        fig.savefig(FIG_DIR / f"decile_returns_{model_name}.png", dpi=150, bbox_inches="tight")
        logger.info(f"Saved figures/decile_returns_{model_name}.png")

    return fig
