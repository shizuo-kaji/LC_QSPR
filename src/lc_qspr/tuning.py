"""Optuna-based LightGBM hyperparameter tuning."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split


def lightgbm_objective(
    trial,
    features: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    *,
    mode: str = "regression",
    random_state: int = 42,
    iterations: int = 5000,
) -> float:
    """Optuna objective matching the original LightGBM search space."""
    if mode not in {"classification", "regression"}:
        raise ValueError("mode must be 'classification' or 'regression'.")
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise ImportError("LightGBM tuning requires the 'training' extra.") from error
    train_x, validation_x, train_y, validation_y = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=random_state,
        stratify=target if mode == "classification" else None,
    )
    parameters: dict[str, Any] = {
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": random_state,
        "force_col_wise": True,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    if mode == "classification":
        parameters.update(
            objective="binary", metric="binary_logloss", is_unbalance=True
        )
    model = lgb.train(
        parameters,
        lgb.Dataset(train_x, label=train_y),
        num_boost_round=iterations,
    )
    prediction = model.predict(validation_x)
    if mode == "classification":
        return metrics.accuracy_score(validation_y, np.rint(prediction))
    return float(np.mean(np.abs(np.asarray(validation_y) - prediction)))


def tune_lightgbm(
    features: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    *,
    mode: str = "regression",
    trials: int = 100,
    random_state: int = 42,
    iterations: int = 5000,
    show_progress: bool = True,
):
    """Run an Optuna study and return it for inspection or persistence."""
    try:
        import optuna
    except ImportError as error:
        raise ImportError("Hyperparameter tuning requires the 'training' extra.") from error
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(
        direction="maximize" if mode == "classification" else "minimize",
        sampler=sampler,
    )
    study.optimize(
        lambda trial: lightgbm_objective(
            trial,
            features,
            target,
            mode=mode,
            random_state=random_state,
            iterations=iterations,
        ),
        n_trials=trials,
        show_progress_bar=show_progress,
    )
    return study

