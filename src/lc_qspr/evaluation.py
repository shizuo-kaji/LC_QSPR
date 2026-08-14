"""Cross-validation, metrics and result-table utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold


def cross_validation_splits(
    table: pd.DataFrame,
    targets: Sequence[str],
    *,
    mode: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic stratified or ordinary K-fold index pairs."""
    if mode not in {"classification", "regression"}:
        raise ValueError("mode must be 'classification' or 'regression'.")
    if mode == "classification" and len(targets) == 1:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        iterator = splitter.split(table, table[targets[0]])
    elif mode == "classification":
        try:
            from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        except ImportError as error:
            raise ImportError(
                "Multi-label splitting requires iterative-stratification."
            ) from error
        splitter = MultilabelStratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        iterator = splitter.split(table, table.loc[:, targets])
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        iterator = splitter.split(table)
    return [(train, test) for train, test in iterator]


def classification_summary(
    results: pd.DataFrame,
    targets: Sequence[str],
    *,
    truth_prefix: str = "true_",
    prediction_prefix: str = "pred_",
) -> pd.DataFrame:
    """Summarise per-target multi-class and positive-versus-negative performance."""
    rows = []
    for target in targets:
        truth = results[f"{truth_prefix}{target}"]
        prediction = results[f"{prediction_prefix}{target}"]
        if truth.nunique(dropna=True) < 2:
            continue
        positive_truth = truth > 0
        positive_prediction = prediction > 0
        row = {
            "type": target.removesuffix("type"),
            "npos": int(positive_truth.sum()),
            "N": len(results),
            "classes": int(truth.nunique()),
            "multiclass_accuracy": metrics.accuracy_score(truth, prediction),
            "cohen_kappa": metrics.cohen_kappa_score(truth, prediction),
            "matthews_correlation": metrics.matthews_corrcoef(truth, prediction),
            "accuracy": metrics.accuracy_score(positive_truth, positive_prediction),
            "recall": metrics.recall_score(
                positive_truth, positive_prediction, zero_division=0
            ),
            "precision": metrics.precision_score(
                positive_truth, positive_prediction, zero_division=0
            ),
            "f1": metrics.f1_score(
                positive_truth, positive_prediction, zero_division=0
            ),
        }
        confidence_column = f"pos_conf_{target}"
        if confidence_column in results and positive_truth.nunique() == 2:
            row["auc"] = metrics.roc_auc_score(
                positive_truth, results[confidence_column]
            )
        error_prefix = f"error_{target.removesuffix('type')}"
        for suffix in ("p", "m"):
            error_column = f"{error_prefix}{suffix}"
            if error_column in results:
                row[f"T{suffix}_mean"] = results[error_column].mean()
                row[f"T{suffix}_std"] = results[error_column].std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("type") if rows else pd.DataFrame()


def classification_details(
    results: pd.DataFrame,
    target: str,
    *,
    truth_prefix: str = "true_",
    prediction_prefix: str = "pred_",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a full classification report and labelled confusion matrix."""
    truth = results[f"{truth_prefix}{target}"]
    prediction = results[f"{prediction_prefix}{target}"]
    labels = sorted(set(truth.dropna()) | set(prediction.dropna()))
    report = pd.DataFrame(
        metrics.classification_report(
            truth,
            prediction,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    confusion = pd.DataFrame(
        metrics.confusion_matrix(truth, prediction, labels=labels),
        index=pd.Index(labels, name="true"),
        columns=pd.Index(labels, name="predicted"),
    )
    return report, confusion


def temperature_summary(
    results: pd.DataFrame, targets: Sequence[str]
) -> pd.DataFrame:
    """Summarise absolute temperature errors by target."""
    rows = []
    for target in targets:
        column = f"error_{target}"
        if column not in results:
            continue
        errors = results[column].dropna()
        rows.append(
            {
                "target": target,
                "n": len(errors),
                "mae": errors.mean(),
                "standard_deviation": errors.std(),
                "median_absolute_error": errors.median(),
            }
        )
    return pd.DataFrame(rows).set_index("target") if rows else pd.DataFrame()


def combine_phase_and_temperature_results(
    phase_results: pd.DataFrame,
    temperature_results: pd.DataFrame,
    *,
    reconstruct_phase_sequence: bool = False,
) -> pd.DataFrame:
    """Join predictions from separate models on their index without duplicate metadata."""
    duplicate_metadata = [
        column for column in ("SMILES", "Phases") if column in temperature_results
    ]
    combined = pd.concat(
        [phase_results, temperature_results.drop(columns=duplicate_metadata)], axis=1
    )
    if reconstruct_phase_sequence:
        from .phases import format_phase_sequence

        combined.insert(
            0,
            "predicted_phase_sequence",
            [format_phase_sequence(row) for row in combined.to_dict(orient="records")],
        )
    return combined
