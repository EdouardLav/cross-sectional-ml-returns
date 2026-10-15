import argparse
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

from config import feature_cfg, wf_cfg
from src.data_loader import run_data_pipeline, get_sp100_tickers
from src.features import build_features, features_to_panel
from src.walk_forward import walk_forward
from src.transaction_costs import run_transaction_cost_analysis
from src.risk_analysis import run_risk_analysis
from src.evaluation import evaluate_all
from src.utils import (
    plot_ic_timeseries,
    plot_cumulative_returns,
    plot_feature_importance,
    plot_ic_comparison_bar,
    plot_decile_returns,
)

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def setup_logging():
    """Configure logging to console and file."""
    log_format = "%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log", mode="w"),
        ],
    )


def main(skip_data: bool = False):
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("CROSS-SECTIONAL ML RETURN PREDICTION PIPELINE")
    logger.info("=" * 70)

    # ──────────────────────────────────────────────
    # STEP 1: DATA
    # ──────────────────────────────────────────────
    data_dir = Path("data")

    if skip_data and (data_dir / "returns.parquet").exists():
        logger.info("Loading cached data from Parquet files...")
        returns = pd.read_parquet(data_dir / "returns.parquet")
        prices = pd.read_parquet(data_dir / "prices.parquet")
        french = pd.read_parquet(data_dir / "french_factors.parquet")
    else:
        returns, prices, french = run_data_pipeline()

    logger.info(f"Returns: {returns.shape} | French: {french.shape}")

    # ──────────────────────────────────────────────
    # STEP 2: FEATURES
    # ──────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("=" * 70)

    # Extract volume data if available
    volume = None
    if isinstance(prices.columns, pd.MultiIndex) and "Volume" in prices.columns.get_level_values(0):
        volume = prices["Volume"]
        # Align volume index with returns
        common_idx = returns.index.intersection(volume.index)
        volume = volume.loc[common_idx]
        returns_aligned = returns.loc[common_idx]
    else:
        returns_aligned = returns

    # Market caps: DISABLED — static snapshot creates look-ahead bias.
    # Today's market cap reflects performance over the test period (e.g. NVDA's
    # current cap embeds its 2015-2024 rally), so the feature leaks future info.
    # With CRSP/Compustat, use point-in-time shares_outstanding_t * price_t.
    market_caps = None

    # Build features
    features = build_features(returns_aligned, volume=volume, market_caps=market_caps)

    # Determine which features were actually computed
    feature_names = list(features.keys())
    logger.info(f"Features computed: {feature_names}")

    # Convert to panel
    panel = features_to_panel(features, returns_aligned)

    # Save panel
    panel.to_parquet(data_dir / "panel.parquet")
    logger.info(f"Panel saved: {panel.shape}")

    # ──────────────────────────────────────────────
    # STEP 3: WALK-FORWARD
    # ──────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: WALK-FORWARD CROSS-VALIDATION")
    logger.info("=" * 70)

    wf_results = walk_forward(panel, feature_names=feature_names)

    predictions = wf_results["predictions"]
    predictions.to_parquet(data_dir / "predictions.parquet")
    logger.info(f"Predictions saved: {predictions.shape}")

    # ──────────────────────────────────────────────
    # STEP 4: EVALUATION
    # ──────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: EVALUATION")
    logger.info("=" * 70)

    eval_results = evaluate_all(predictions, french_factors=french)

    # Save evaluation tables
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    for name, df in eval_results.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(results_dir / f"{name}.csv")
            logger.info(f"Saved results/{name}.csv")

    # ──────────────────────────────────────────────
    # STEP 5: FIGURES
    # ──────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: GENERATING FIGURES")
    logger.info("=" * 70)

    model_names = [
        col for col in predictions.columns
        if col not in ("date", "ticker", "realized")
    ]

    # IC time series
    plot_ic_timeseries(eval_results["ic_monthly"], model_names)

    # Cumulative returns
    plot_cumulative_returns(eval_results["portfolio_returns"], model_names)

    # Feature importance
    plot_feature_importance(wf_results["feature_importance"], model_names)

    # IC comparison
    plot_ic_comparison_bar(eval_results["ic_summary"])

    # Decile returns for best model (by Sharpe)
    best_model = eval_results["portfolio_summary"]["sharpe"].idxmax()
    plot_decile_returns(predictions, best_model)

    # ──────────────────────────────────────────────
    # STEP 6: TRANSACTION COST ANALYSIS
    # ──────────────────────────────────────────────

    tc_results = run_transaction_cost_analysis(predictions, eval_results["portfolio_returns"])

    # ── STEP 7: RISK & ROBUSTNESS ANALYSIS ──
    risk_results = run_risk_analysis(
        predictions,
        eval_results["portfolio_returns"],
        french,
        headline_model="Ridge",
    )

    # Persist the hyperparameters selected by the inner walk-forward
    if wf_results.get("param_history"):
        from src.nested_cv import summarize_param_history
        summarize_param_history(wf_results["param_history"]).to_csv(
            results_dir / "selected_hyperparams.csv", index=False
        )
        logger.info("Saved results/selected_hyperparams.csv")

    # ──────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE — SUMMARY")
    logger.info("=" * 70)
    logger.info(f"\nIC Summary:\n{eval_results['ic_summary']}")
    logger.info(f"\nPortfolio Summary:\n{eval_results['portfolio_summary']}")
    if "factor_alpha" in eval_results:
        logger.info(f"\nFactor Alpha:\n{eval_results['factor_alpha']}")

    logger.info("\nAll results saved to results/ and figures/")
    logger.info("Run notebooks for interactive exploration.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Sectional ML Return Prediction")
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data download, use cached Parquet files",
    )
    args = parser.parse_args()
    main(skip_data=args.skip_data)
