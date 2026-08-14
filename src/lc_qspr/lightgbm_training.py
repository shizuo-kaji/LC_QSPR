"""Cross-validated LightGBM training and feature-importance utilities."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .evaluation import classification_summary, temperature_summary


@dataclass(frozen=True)
class LightGBMConfig:
    """LightGBM configuration shared by classification and regression."""

    iterations: int = 10000
    learning_rate: float = 0.01
    random_seed: int = 42
    num_threads: int = -1
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class LightGBMCVResult:
    models: dict[str, list[Any]]
    predictions: pd.DataFrame
    summary: pd.DataFrame
    evaluation_history: dict[str, list[dict]]
    feature_importance: dict[str, pd.DataFrame]


def lightgbm_accuracy(predictions, dataset):
    """LightGBM custom multi-class accuracy metric."""
    truth = dataset.get_label()
    predictions = np.asarray(predictions)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, len(truth)).T
    return "accuracy", (truth == np.argmax(predictions, axis=1)).mean(), True


def lightgbm_mae(predictions, dataset):
    """LightGBM custom mean absolute error metric."""
    truth = dataset.get_label()
    return "MAE", np.nanmean(np.abs(truth - predictions)), False


class LightGBMProgressBar:
    """Optional tqdm callback for long LightGBM fits."""

    order = 20
    before_iteration = False

    def __init__(self, description: str = "LightGBM", update_interval: int = 10):
        from tqdm.auto import tqdm

        self.description = description
        self.update_interval = update_interval
        self.progress = tqdm(disable=None)

    def __call__(self, environment):
        if environment.iteration == environment.begin_iteration:
            self.progress.reset(total=environment.end_iteration - environment.begin_iteration)
            self.progress.set_description(self.description, refresh=False)
        if (
            environment.iteration - environment.begin_iteration
        ) % self.update_interval == 0 and environment.evaluation_result_list:
            self.progress.set_postfix(
                OrderedDict(
                    (f"{entry[0]}:{entry[1]}", f"{entry[2]:.5f}")
                    for entry in environment.evaluation_result_list
                ),
                refresh=False,
            )
        self.progress.update(1)


def lightgbm_feature_importance(
    models: Sequence[Any],
    feature_names: Sequence[str],
    descriptor_families: dict[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Combine fold-wise importance and optionally tag physical families."""
    importance = pd.concat(
        [
            pd.Series(model.feature_importance(), index=feature_names, name=f"CV{index}")
            for index, model in enumerate(models)
        ],
        axis=1,
    )
    importance["median"] = importance.median(axis=1)
    if descriptor_families:
        physical = {
            family: set(names)
            for family, names in descriptor_families.items()
            if family in {"Shape", "Electric", "Flexible"}
        }
        importance["family"] = [
            next((family for family, names in physical.items() if name in names), "Other")
            for name in importance.index
        ]
    return importance.sort_values("median", ascending=False)


def _lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise ImportError("LightGBM training requires the 'training' extra.") from error
    return lgb


def train_lightgbm_phases(
    table: pd.DataFrame,
    feature_names: Sequence[str],
    targets: Sequence[str],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    config: LightGBMConfig | None = None,
    output_directory: str | Path | None = None,
    show_progress: bool = False,
    families: dict[str, Sequence[str]] | None = None,
) -> LightGBMCVResult:
    """Train one multi-class-OVA LightGBM model per phase target and fold."""
    lgb = _lightgbm()
    config = config or LightGBMConfig()
    features = table.loc[:, list(feature_names)].to_numpy()
    result = table.loc[:, [c for c in ("SMILES", "Phases") if c in table]].copy()
    models: dict[str, list[Any]] = {}
    histories: dict[str, list[dict]] = {}
    importances = {}
    base_parameters = {
        "learning_rate": config.learning_rate,
        "boosting_type": "gbdt",
        "objective": "multiclassova",
        "is_unbalance": True,
        "max_depth": 10,
        "num_leaves": 256,
        "seed": config.random_seed,
        "force_col_wise": True,
        "min_child_samples": 20,
        "lambda_l2": 0.327,
        "feature_fraction": 0.775,
        "bagging_fraction": 0.643,
        "num_threads": config.num_threads,
        "verbosity": -1,
        **config.parameters,
    }
    for target in targets:
        labels = table[target].to_numpy()
        number_of_classes = max(len(np.unique(labels)), 2)
        probabilities = np.zeros((len(labels), number_of_classes), dtype=float)
        models[target] = []
        histories[target] = []
        for fold, (train_indices, test_indices) in enumerate(splits):
            training = lgb.Dataset(features[train_indices], label=labels[train_indices])
            validation = lgb.Dataset(features[test_indices], label=labels[test_indices])
            history: dict = {}
            callbacks = [lgb.record_evaluation(history)]
            if show_progress:
                callbacks.append(LightGBMProgressBar(f"{target} CV{fold}"))
            model = lgb.train(
                {**base_parameters, "num_class": number_of_classes},
                training,
                num_boost_round=config.iterations,
                valid_names=["validation"],
                valid_sets=[validation],
                feval=lightgbm_accuracy,
                callbacks=callbacks,
            )
            probabilities[test_indices] = model.predict(features[test_indices])
            models[target].append(model)
            histories[target].append(history)
        result[f"true_{target}"] = labels
        result[f"pred_{target}"] = probabilities.argmax(axis=1)
        result[f"conf_{target}"] = probabilities.max(axis=1)
        result[f"pos_conf_{target}"] = probabilities[:, 1:].sum(axis=1)
        importances[target] = lightgbm_feature_importance(
            models[target], feature_names, families
        )
        if output_directory is not None:
            directory = Path(output_directory)
            directory.mkdir(parents=True, exist_ok=True)
            models[target][0].save_model(directory / f"LightGBM_{target}.txt")
            importances[target].to_csv(directory / f"importance_{target}.csv")
    return LightGBMCVResult(
        models,
        result,
        classification_summary(result, targets),
        histories,
        importances,
    )


def train_lightgbm_temperatures(
    table: pd.DataFrame,
    feature_names: Sequence[str],
    targets: Sequence[str],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    config: LightGBMConfig | None = None,
    output_directory: str | Path | None = None,
    show_progress: bool = False,
    families: dict[str, Sequence[str]] | None = None,
) -> LightGBMCVResult:
    """Train one missing-target-aware LightGBM regressor per temperature target."""
    lgb = _lightgbm()
    config = config or LightGBMConfig(iterations=50000)
    features = table.loc[:, list(feature_names)].to_numpy()
    result = table.loc[:, [c for c in ("SMILES", "Phases") if c in table]].copy()
    models: dict[str, list[Any]] = {}
    histories: dict[str, list[dict]] = {}
    importances = {}
    base_parameters = {
        "learning_rate": config.learning_rate,
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "l2",
        "seed": config.random_seed,
        "force_col_wise": True,
        "min_data": 20,
        "max_depth": 10,
        "num_leaves": 32,
        "lambda_l2": 0.327,
        "feature_fraction": 0.494,
        "bagging_fraction": 0.845,
        "bagging_freq": 3,
        "num_threads": config.num_threads,
        "verbosity": -1,
        **config.parameters,
    }
    for target in targets:
        labels = table[target].to_numpy(dtype=float)
        predictions = np.full(len(labels), np.nan)
        models[target] = []
        histories[target] = []
        for fold, (train_indices, test_indices) in enumerate(splits):
            train_observed = ~np.isnan(labels[train_indices])
            test_observed = ~np.isnan(labels[test_indices])
            if not train_observed.any():
                raise ValueError(f"No observed training values for {target} in fold {fold}.")
            training = lgb.Dataset(
                features[train_indices][train_observed],
                label=labels[train_indices][train_observed],
            )
            history: dict = {}
            callbacks = [lgb.record_evaluation(history)]
            if show_progress:
                callbacks.append(LightGBMProgressBar(f"{target} CV{fold}"))
            train_kwargs = {}
            if test_observed.any():
                validation = lgb.Dataset(
                    features[test_indices][test_observed],
                    label=labels[test_indices][test_observed],
                )
                train_kwargs = {
                    "valid_names": ["validation"],
                    "valid_sets": [validation],
                    "feval": lightgbm_mae,
                }
            model = lgb.train(
                base_parameters,
                training,
                num_boost_round=config.iterations,
                callbacks=callbacks,
                **train_kwargs,
            )
            predictions[test_indices] = model.predict(features[test_indices])
            models[target].append(model)
            histories[target].append(history)
        result[f"true_{target}"] = labels
        result[f"pred_{target}"] = predictions
        result[f"error_{target}"] = np.abs(predictions - labels)
        importances[target] = lightgbm_feature_importance(
            models[target], feature_names, families
        )
        if output_directory is not None:
            directory = Path(output_directory)
            directory.mkdir(parents=True, exist_ok=True)
            models[target][0].save_model(directory / f"LightGBM_{target}.txt")
            importances[target].to_csv(directory / f"importance_{target}.csv")
    return LightGBMCVResult(
        models,
        result,
        temperature_summary(result, targets),
        histories,
        importances,
    )

