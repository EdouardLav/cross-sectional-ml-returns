"""
Transaction Cost Analysis
=========================
Computes:
  1. Monthly portfolio composition (which tickers in long/short)
  2. Turnover between consecutive months
  3. Net-of-cost Sharpe ratios
  4. Break-even cost analysis (Sharpe vs cost in bps)

This module uses the existing predictions DataFrame from walk_forward.py.
No changes needed to the rest of the pipeline.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import eval_cfg

logger = logging.getLogger(__name__)

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

COLORS = ["#2c3e50", "#e74c3c", "#27ae60", "#f39c12"]


# ──────────────────────────────────────────────────
# 1. PORTFOLIO COMPOSITION AT EACH DATE
# ──────────────────────────────────────────────────

def get_portfolio_positions(
    predictions: pd.DataFrame,
    model_name: str,
    n_quantiles: int = eval_cfg.n_quantiles,
    long_q: int = eval_cfg.long_quantile,
    short_q: int = eval_cfg.short_quantile,
) -> dict:
    """
    For each date, identify which tickers are in the long and short portfolios.

    Returns
    -------
    positions : dict
        {date: {"long": set(tickers), "short": set(tickers)}}
    """
    positions = {}

    for date, group in predictions.groupby("date"):
        if len(group) < n_quantiles:
            continue

        group = group.copy()
        group["quantile"] = pd.qcut(
            group[model_name].rank(method="first"),
            q=n_quantiles,
            labels=range(1, n_quantiles + 1),
        )

        long_tickers = set(group.loc[group["quantile"] == long_q, "ticker"])
        short_tickers = set(group.loc[group["quantile"] == short_q, "ticker"])

        positions[date] = {"long": long_tickers, "short": short_tickers}

    return positions


# ──────────────────────────────────────────────────
# 2. TURNOVER COMPUTATION
# ──────────────────────────────────────────────────

def compute_turnover(positions: dict) -> pd.Series:
    """
    Compute monthly one-way turnover from portfolio positions.

    Turnover at date t = fraction of the portfolio that changed since t-1.
    Computed as: (entries + exits) / (2 * portfolio_size) for each side,
    then averaged across long and short.

    A turnover of 0.5 means 50% of positions changed — typical for
    monthly decile sorts on ~100 stocks.
    """
    dates = sorted(positions.keys())
    turnover = {}

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]

        prev = positions[prev_date]
        curr = positions[curr_date]

        # Long side turnover
        long_entries = curr["long"] - prev["long"]
        long_exits = prev["long"] - curr["long"]
        long_size = max(len(curr["long"]), 1)
        long_turnover = len(long_entries) / long_size

        # Short side turnover
        short_entries = curr["short"] - prev["short"]
        short_exits = prev["short"] - curr["short"]
        short_size = max(len(curr["short"]), 1)
        short_turnover = len(short_entries) / short_size

        # Average one-way turnover across both sides
        turnover[curr_date] = (long_turnover + short_turnover) / 2

    return pd.Series(turnover, name="turnover").sort_index()


# ──────────────────────────────────────────────────
# 3. NET-OF-COST RETURNS AND SHARPE
# ──────────────────────────────────────────────────

def compute_net_returns(
    gross_returns: pd.Series,
    turnover: pd.Series,
    cost_bps_oneway: float = 15.0,
) -> pd.Series:
    """
    Subtract transaction costs from gross long-short returns.

    Net return_t = gross_return_t - turnover_t * 2 * cost_bps / 10000

    The factor of 2 converts one-way cost to round-trip:
    each entry requires a buy (cost) and each exit requires a sell (cost).

    Parameters
    ----------
    gross_returns : pd.Series
        Monthly gross long-short returns.
    turnover : pd.Series
        Monthly one-way turnover (fraction of portfolio that changed).
    cost_bps_oneway : float
        One-way transaction cost in basis points (default: 15 bps).
    """
    # Align dates
    common = gross_returns.index.intersection(turnover.index)
    gross = gross_returns.loc[common]
    to = turnover.loc[common]

    # Round-trip cost = 2 * one-way cost
    cost_per_unit = 2 * cost_bps_oneway / 10000.0
    trading_cost = to * cost_per_unit

    net = gross - trading_cost

    return net


def compute_sharpe(returns: pd.Series, annualization: int = 12) -> float:
    """Annualized Sharpe ratio from monthly returns."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return (returns.mean() * annualization) / (returns.std() * np.sqrt(annualization))


# ──────────────────────────────────────────────────
# 4. BREAK-EVEN ANALYSIS
# ──────────────────────────────────────────────────

def break_even_analysis(
    predictions: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    model_names: list[str],
    max_cost_bps: int = 100,
    step_bps: int = 5,
) -> dict:
    """
    For each model, compute Sharpe ratio as a function of transaction cost.

    Returns
    -------
    results : dict
        {
            "cost_grid": list of cost values in bps,
            "sharpe_curves": {model_name: list of Sharpe values},
            "turnover_stats": {model_name: {"mean": ..., "median": ..., "std": ...}},
            "net_sharpe_15bps": {model_name: float},
            "break_even_bps": {model_name: float or None},
        }
    """
    cost_grid = list(range(0, max_cost_bps + 1, step_bps))
    sharpe_curves = {}
    turnover_stats = {}
    net_sharpe_15bps = {}
    break_even_bps = {}

    for model_name in model_names:
        logger.info(f"\nAnalyzing transaction costs for {model_name}...")

        # Get positions and turnover
        positions = get_portfolio_positions(predictions, model_name)
        turnover = compute_turnover(positions)

        turnover_stats[model_name] = {
            "mean": turnover.mean(),
            "median": turnover.median(),
            "std": turnover.std(),
        }
        logger.info(
            f"  Turnover: mean={turnover.mean():.3f}, "
            f"median={turnover.median():.3f}, std={turnover.std():.3f}"
        )

        # Gross returns
        ls_col = f"{model_name}_long_short"
        if ls_col not in portfolio_returns.columns:
            continue
        gross = portfolio_returns[ls_col].dropna()

        # Compute Sharpe for each cost level
        sharpes = []
        for cost in cost_grid:
            net = compute_net_returns(gross, turnover, cost_bps_oneway=cost)
            s = compute_sharpe(net)
            sharpes.append(s)
        sharpe_curves[model_name] = sharpes

        # Net Sharpe at 15 bps one-way (30 bps round-trip)
        net_15 = compute_net_returns(gross, turnover, cost_bps_oneway=15)
        net_sharpe_15bps[model_name] = compute_sharpe(net_15)
        logger.info(f"  Sharpe gross: {compute_sharpe(gross):.3f}")
        logger.info(f"  Sharpe net (15bps OW): {net_sharpe_15bps[model_name]:.3f}")

        # Break-even: find cost where Sharpe crosses zero
        be = None
        for i in range(len(sharpes) - 1):
            if sharpes[i] >= 0 and sharpes[i + 1] < 0:
                # Linear interpolation
                c1, c2 = cost_grid[i], cost_grid[i + 1]
                s1, s2 = sharpes[i], sharpes[i + 1]
                be = c1 + (0 - s1) * (c2 - c1) / (s2 - s1)
                break
        if be is None and sharpes[-1] > 0:
            be = float("inf")  # still profitable at max cost
        break_even_bps[model_name] = be
        logger.info(f"  Break-even cost: {be:.0f} bps one-way" if be and be != float("inf") else f"  Break-even: > {max_cost_bps} bps")

    return {
        "cost_grid": cost_grid,
        "sharpe_curves": sharpe_curves,
        "turnover_stats": turnover_stats,
        "net_sharpe_15bps": net_sharpe_15bps,
        "break_even_bps": break_even_bps,
    }


# ──────────────────────────────────────────────────
# 5. PLOTS
# ──────────────────────────────────────────────────

def plot_break_even(results: dict, save: bool = True):
    """
    Plot Sharpe ratio vs transaction cost for each model.

    This is the money plot for a recruiter: shows at what cost level
    the strategy's edge disappears.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    cost_grid = results["cost_grid"]
    model_names = list(results["sharpe_curves"].keys())

    for model_name, color in zip(model_names, COLORS):
        sharpes = results["sharpe_curves"][model_name]
        ax.plot(cost_grid, sharpes, color=color, linewidth=2.5, label=model_name, marker="o", markersize=4)

        # Mark break-even point
        be = results["break_even_bps"].get(model_name)
        if be and be != float("inf") and be <= max(cost_grid):
            ax.plot(be, 0, marker="x", color=color, markersize=12, markeredgewidth=3)
            ax.annotate(
                f"{be:.0f} bps",
                (be, 0),
                textcoords="offset points",
                xytext=(10, 10),
                fontsize=10,
                color=color,
                fontweight="bold",
            )

    # Zero line
    ax.axhline(y=0, color="black", linewidth=1, linestyle="--", alpha=0.5)

    # Reference line at 15 bps
    ax.axvline(x=15, color="grey", linewidth=1, linestyle=":", alpha=0.7)
    ax.annotate(
        "15 bps\n(S&P 100 est.)",
        (15, ax.get_ylim()[1] * 0.85),
        fontsize=9,
        color="grey",
        ha="center",
    )

    ax.set_xlabel("One-Way Transaction Cost (bps)", fontsize=12)
    ax.set_ylabel("Annualized Sharpe Ratio", fontsize=12)
    ax.set_title("Break-Even Analysis: Sharpe Ratio vs Transaction Costs", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        fig.savefig(FIG_DIR / "break_even_analysis.png", dpi=150, bbox_inches="tight")
        logger.info("Saved figures/break_even_analysis.png")

    return fig


def plot_turnover_timeseries(
    predictions: pd.DataFrame,
    model_names: list[str],
    save: bool = True,
):
    """Plot monthly turnover over time for each model."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for model_name, color in zip(model_names, COLORS):
        positions = get_portfolio_positions(predictions, model_name)
        turnover = compute_turnover(positions)
        ax.plot(turnover.index, turnover.values, color=color, alpha=0.7, linewidth=1.5, label=model_name)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("One-Way Turnover (fraction)", fontsize=12)
    ax.set_title("Monthly Portfolio Turnover by Model", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        fig.savefig(FIG_DIR / "turnover_timeseries.png", dpi=150, bbox_inches="tight")
        logger.info("Saved figures/turnover_timeseries.png")

    return fig


# ──────────────────────────────────────────────────
# 6. MAIN ENTRY POINT
# ──────────────────────────────────────────────────

def run_transaction_cost_analysis(
    predictions: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
) -> dict:
    """
    Run the full transaction cost analysis.

    Call this after evaluate_all() in main.py.
    """
    model_names = [
        col for col in predictions.columns
        if col not in ("date", "ticker", "realized")
    ]

    logger.info("=" * 60)
    logger.info("TRANSACTION COST ANALYSIS")
    logger.info("=" * 60)

    results = break_even_analysis(predictions, portfolio_returns, model_names)

    # Summary table
    logger.info("\n" + "=" * 60)
    logger.info("TRANSACTION COST SUMMARY")
    logger.info("=" * 60)
    for model in model_names:
        to = results["turnover_stats"].get(model, {})
        net_s = results["net_sharpe_15bps"].get(model, 0)
        be = results["break_even_bps"].get(model)
        be_str = f"{be:.0f}" if be and be != float("inf") else ">100"
        logger.info(
            f"  {model:15s} | Turnover: {to.get('mean', 0):.1%} | "
            f"Sharpe net (15bps): {net_s:.3f} | "
            f"Break-even: {be_str} bps"
        )

    # Plots
    plot_break_even(results)
    plot_turnover_timeseries(predictions, model_names)

    # Save results
    summary = pd.DataFrame({
        "model": model_names,
        "mean_turnover": [results["turnover_stats"].get(m, {}).get("mean", 0) for m in model_names],
        "sharpe_gross": [results["sharpe_curves"][m][0] if m in results["sharpe_curves"] else 0 for m in model_names],
        "sharpe_net_15bps": [results["net_sharpe_15bps"].get(m, 0) for m in model_names],
        "break_even_bps": [results["break_even_bps"].get(m) for m in model_names],
    }).set_index("model")
    summary.to_csv("results/transaction_costs.csv")
    logger.info("Saved results/transaction_costs.csv")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    # Load saved data
    predictions = pd.read_parquet("data/predictions.parquet")
    portfolio_returns = pd.read_csv("results/portfolio_returns.csv", index_col="date", parse_dates=True)

    results = run_transaction_cost_analysis(predictions, portfolio_returns)
