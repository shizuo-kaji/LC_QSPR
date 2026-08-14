"""Cross-validated CatBoost training for phase and temperature targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from .evaluation import classification_summary, temperature_summary


@dataclass(frozen=True)
class CatBoostConfig:
    """Shared CatBoost hyperparameters."""

    iterations: int = 5000
    learning_rate: float = 0.3
    l2_leaf_reg: float = 1.0
    random_seed: int = 42
    thread_count: int = -1
    verbose: int | bool = 500
    extra_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class CatBoostCVResult:
    """Models, out-of-fold predictions and summary from one CV run."""

    models: list[Any]
    predictions: pd.DataFrame
    summary: pd.DataFrame
    feature_names: list[str]


def train_catboost_phases(
    table: pd.DataFrame,
    feature_names: Sequence[str],
    targets: Sequence[str],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    config: CatBoostConfig | None = None,
    model_path: str | Path | None = None,
) -> CatBoostCVResult:
    """Train a multi-label phase classifier and collect out-of-fold probabilities."""
    config = config or CatBoostConfig()
    feature_names = list(feature_names)
    targets = list(targets)
    features = table.loc[:, feature_names]
    labels = table.loc[:, targets]
    probabilities = np.zeros(labels.shape, dtype=float)
    models = []
    for train_indices, test_indices in splits:
        model = CatBoostClassifier(
            iterations=config.iterations,
            learning_rate=config.learning_rate,
            l2_leaf_reg=config.l2_leaf_reg,
            random_seed=config.random_seed,
            thread_count=config.thread_count,
            verbose=config.verbose,
            loss_function="MultiLogloss",
            eval_metric="Accuracy",
            **config.extra_parameters,
        )
        model.fit(
            features.iloc[train_indices],
            labels.iloc[train_indices],
            eval_set=(features.iloc[test_indices], labels.iloc[test_indices]),
        )
        probabilities[test_indices] = np.asarray(
            model.predict_proba(features.iloc[test_indices])
        )
        models.append(model)

    predictions = table.loc[:, [c for c in ("SMILES", "Phases") if c in table]].copy()
    for target_index, target in enumerate(targets):
        predictions[f"true_{target}"] = labels[target].to_numpy()
        predictions[f"pred_{target}"] = (
            probabilities[:, target_index] >= 0.5
        ).astype(int)
        predictions[f"conf_{target}"] = np.maximum(
            probabilities[:, target_index], 1 - probabilities[:, target_index]
        )
        predictions[f"pos_conf_{target}"] = probabilities[:, target_index]
    summary = classification_summary(predictions, targets)
    if model_path is not None:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        models[-1].save_model(path, format="cbm")
    return CatBoostCVResult(models, predictions, summary, feature_names)


def train_catboost_temperatures(
    table: pd.DataFrame,
    feature_names: Sequence[str],
    targets: Sequence[str],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    config: CatBoostConfig | None = None,
    model_path: str | Path | None = None,
) -> CatBoostCVResult:
    """Train the multi-output regressor with missing-value-aware MultiRMSE."""
    config = config or CatBoostConfig(iterations=10000)
    feature_names = list(feature_names)
    targets = list(targets)
    features = table.loc[:, feature_names]
    labels = table.loc[:, targets]
    predictions_array = np.full(labels.shape, np.nan, dtype=float)
    models = []
    for train_indices, test_indices in splits:
        model = CatBoostRegressor(
            iterations=config.iterations,
            learning_rate=config.learning_rate,
            l2_leaf_reg=config.l2_leaf_reg,
            random_seed=config.random_seed,
            thread_count=config.thread_count,
            verbose=config.verbose,
            loss_function="MultiRMSEWithMissingValues",
            **config.extra_parameters,
        )
        model.fit(
            features.iloc[train_indices],
            labels.iloc[train_indices],
            eval_set=(features.iloc[test_indices], labels.iloc[test_indices]),
        )
        predictions_array[test_indices] = np.asarray(
            model.predict(features.iloc[test_indices])
        )
        models.append(model)

    predictions = table.loc[:, [c for c in ("SMILES", "Phases") if c in table]].copy()
    for target_index, target in enumerate(targets):
        truth = labels[target].to_numpy()
        prediction = predictions_array[:, target_index]
        predictions[f"true_{target}"] = truth
        predictions[f"pred_{target}"] = prediction
        predictions[f"error_{target}"] = np.abs(prediction - truth)
    summary = temperature_summary(predictions, targets)
    if model_path is not None:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        models[-1].save_model(path, format="cbm")
    return CatBoostCVResult(models, predictions, summary, feature_names)


def catboost_feature_importance(
    model: Any, feature_names: Sequence[str], *, top_n: int | None = 100
) -> pd.Series:
    """Return sorted feature importance from a fitted CatBoost model."""
    importance = pd.Series(model.feature_importances_, index=list(feature_names))
    importance = importance.sort_values(ascending=False)
    return importance if top_n is None else importance.head(top_n)


def catboost_shap_values(model: Any, features: pd.DataFrame):
    """Calculate SHAP values, importing SHAP only when requested."""
    try:
        import shap
    except ImportError as error:
        raise ImportError("SHAP analysis requires the 'training' extra.") from error
    return shap.TreeExplainer(model).shap_values(features)

