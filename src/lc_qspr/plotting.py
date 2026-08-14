"""Plots used by the training and interpretation notebooks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_catboost_importance(importance: pd.Series, *, title: str = "CatBoost feature importance"):
    """Plot a sorted horizontal CatBoost importance chart."""
    figure, axis = plt.subplots(figsize=(7, max(4, 0.22 * len(importance))))
    importance.sort_values().plot.barh(ax=axis, color="turquoise")
    axis.set_xlabel("Feature importance")
    axis.set_title(title)
    figure.tight_layout()
    return figure, axis


def plot_lightgbm_histories(
    histories: dict[str, list[dict]], *, metric_names: Sequence[str] | None = None
):
    """Plot available validation metrics for every LightGBM target and fold."""
    targets = list(histories)
    figure, axes = plt.subplots(len(targets), 1, figsize=(8, 3 * len(targets)), squeeze=False)
    for target_index, target in enumerate(targets):
        axis = axes[target_index, 0]
        for fold, history in enumerate(histories[target]):
            for dataset_metrics in history.values():
                for metric, values in dataset_metrics.items():
                    if metric_names is None or metric in metric_names:
                        axis.plot(values, label=f"CV{fold} {metric}")
        axis.set_title(target)
        axis.set_xlabel("Iteration")
        axis.legend()
    figure.tight_layout()
    return figure, axes


def plot_gnn_histories(
    histories: Sequence[dict[str, list[Any]]],
    *,
    metric: str = "loss",
    target_index: int = 0,
):
    """Plot fold-wise training and validation curves for one GNN metric."""
    if metric not in {"loss", "accuracy", "f1", "mae"}:
        raise ValueError("metric must be loss, accuracy, f1 or mae.")
    figure, axis = plt.subplots(figsize=(8, 5))
    for fold, history in enumerate(histories):
        for split, colour in (("training", "tab:cyan"), ("validation", "tab:green")):
            if metric == "loss":
                values = [entry.loss for entry in history[split]]
            else:
                values = [getattr(entry, metric)[target_index] for entry in history[split]]
            axis.plot(values, color=colour, alpha=0.35, label=f"CV{fold} {split}")
    axis.set_title(f"GNN {metric}")
    axis.set_xlabel("Epoch")
    axis.legend(ncol=2)
    figure.tight_layout()
    return figure, axis
