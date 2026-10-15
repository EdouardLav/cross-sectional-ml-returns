"""
Walk-Forward Cross-Validation
==============================
Expanding window walk-forward to avoid temporal data leakage.

Why NOT standard K-fold cross-validation?
  K-fold randomly splits data into train/test folds. In time-series / panel data,
  this means FUTURE observations can end up in the training set while we predict
  the PAST. Example: with 5-fold CV, data from 2020 could train a model that
  predicts 2015 returns. This is look-ahead bias — the model has seen the future.

  Walk-forward respects the arrow of time:
    - Train on [2006, ..., Y-1]
    - Predict all months of year Y
    - Advance Y by one year, expand the training set
    - Repeat
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import wf_cfg, model_cfg
from src.models import build_models, get_feature_importance
from src.nested_cv import (
    select_hyperparams,
    build_selected,
    plot_hyperparam_trajectory,
    summarize_param_history,
)

logger = logging.getLogger(__name__)


def walk_forward(
    panel: pd.DataFrame,
    feature_names: list[str],
    target_col: str = "forward_return",
    train_start_year: int = wf_cfg.train_start_year,
    test_start_year: int = wf_cfg.test_start_year,
    test_end_year: int = wf_cfg.test_end_year,
    tune: bool = True,
) -> dict:
    """
    Run expanding-window walk-forward cross-validation.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel data with columns: date, ticker, features..., forward_return.
    feature_names : list of str
        Column names of features to use.
    target_col : str
        Name of the target column.
    train_start_year : int
        First year of training data.
    test_start_year : int
        First year of out-of-sample testing.
    test_end_year : int
        Last year of out-of-sample testing.

    Returns
    -------
    results : dict
        {
            'predictions': DataFrame with columns [date, ticker, realized, model_1, ..., model_n],
            'feature_importance': dict of {model_name: {year: {feature: importance}}},
            'training_info': list of dicts with training metadata per fold
        }
    """
    models = build_models()
    model_names = list(models.keys())

    all_predictions = []
    feature_importance = {name: {} for name in model_names}
    training_info = []
    param_history = {name: {} for name in model_names}

    # Extract year from date column
    panel = panel.copy()
    panel["year"] = panel["date"].dt.year

    test_years = range(test_start_year, test_end_year + 1)

    for test_year in test_years:
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD: Train [{train_start_year}-{test_year - 1}] → Test [{test_year}]")
        logger.info(f"{'='*60}")

        # ── Split ──
        train_mask = (panel["year"] >= train_start_year) & (panel["year"] < test_year)
        test_mask = panel["year"] == test_year

        train_data = panel.loc[train_mask]
        test_data = panel.loc[test_mask]

        if len(train_data) == 0 or len(test_data) == 0:
            logger.warning(f"  Skipping {test_year}: empty train or test set")
            continue

        X_train = train_data[feature_names].values
        y_train = train_data[target_col].values
        X_test = test_data[feature_names].values
        y_test = test_data[target_col].values

        logger.info(f"  Train: {len(train_data)} obs, {train_data['year'].min()}-{train_data['year'].max()}")
        logger.info(f"  Test:  {len(test_data)} obs, {test_data['date'].nunique()} months")

        fold_info = {
            "test_year": test_year,
            "n_train": len(train_data),
            "n_test": len(test_data),
            "train_years": f"{train_start_year}-{test_year - 1}",
        }

        # ── Base prediction DataFrame for this fold ──
        fold_preds = test_data[["date", "ticker", target_col]].copy()
        fold_preds = fold_preds.rename(columns={target_col: "realized"})

        # ── Fit and predict each model ──
        for model_name, model in models.items():
            logger.info(f"  Training {model_name}...")

            if tune:
                # Hyperparameters chosen by an inner walk-forward on train_data
                # ONLY. The outer test year never informs the selection.
                best_params, _ = select_hyperparams(
                    train_data, feature_names, model_name, target_col=target_col
                )
                param_history[model_name][test_year] = best_params
                m = build_selected(model_name, best_params)
            else:
                # Re-instantiate to avoid state leakage between folds
                fresh_models = build_models()
                m = fresh_models[model_name]

            m.fit(X_train, y_train)
            preds = m.predict(X_test)

            fold_preds[model_name] = preds

            # Feature importance
            fi = get_feature_importance(m, feature_names)
            feature_importance[model_name][test_year] = fi

            # In-sample R² (diagnostic, NOT for model selection)
            train_preds = m.predict(X_train)
            ss_res = np.sum((y_train - train_preds) ** 2)
            ss_tot = np.sum((y_train - y_train.mean()) ** 2)
            r2_train = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            logger.info(f"    In-sample R²: {r2_train:.4f}")

            fold_info[f"{model_name}_train_r2"] = r2_train

        all_predictions.append(fold_preds)
        training_info.append(fold_info)

    # ── Combine all folds ──
    predictions = pd.concat(all_predictions, ignore_index=True)

    logger.info(f"\nWalk-forward complete:")
    logger.info(f"  Total OOS predictions: {len(predictions)}")
    logger.info(f"  Date range: {predictions['date'].min()} to {predictions['date'].max()}")

    if tune:
        plot_hyperparam_trajectory(param_history)
        params_df = summarize_param_history(param_history)
        logger.info(f"\nSelected hyperparameters per fold:\n{params_df.to_string(index=False)}")

    return {
        "predictions": predictions,
        "feature_importance": feature_importance,
        "training_info": training_info,
        "param_history": param_history,
    }
