"""Descriptor-level statistical comparisons."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from scipy import stats


def mann_whitney_descriptors(
    table: pd.DataFrame,
    label: str,
    descriptor_names: Sequence[str],
) -> pd.DataFrame:
    """Rank descriptors by a two-sided Mann–Whitney U comparison."""
    positive = table[label] > 0
    rows = []
    for descriptor in descriptor_names:
        positive_values = table.loc[positive, descriptor].dropna()
        negative_values = table.loc[~positive, descriptor].dropna()
        if positive_values.empty or negative_values.empty:
            continue
        try:
            statistic, p_value = stats.mannwhitneyu(
                positive_values, negative_values, alternative="two-sided"
            )
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "descriptor": descriptor,
                "u_statistic": statistic,
                "p_value": p_value,
                "positive_n": len(positive_values),
                "negative_n": len(negative_values),
                "positive_median": positive_values.median(),
                "negative_median": negative_values.median(),
            }
        )
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    result["benjamini_hochberg_q"] = _benjamini_hochberg(result["p_value"])
    return result


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    count = len(p_values)
    adjusted = p_values * count / pd.Series(
        range(1, count + 1), index=p_values.index, dtype=float
    )
    return adjusted.iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)


def plot_descriptor_boxplots(
    table: pd.DataFrame,
    label: str,
    descriptor_names: Sequence[str],
    *,
    maximum_descriptors: int = 5,
    annotate: bool = True,
):
    """Plot phase-negative and phase-positive descriptor distributions."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    names = list(descriptor_names)[:maximum_descriptors]
    if not names:
        raise ValueError("At least one descriptor is required.")
    figure, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 6))
    if len(names) == 1:
        axes = [axes]
    for axis, descriptor in zip(axes, names):
        sns.boxplot(data=table, x=label, y=descriptor, ax=axis)
        axis.set_title(descriptor)
        if annotate:
            try:
                from statannotations.Annotator import Annotator
            except ImportError as error:
                raise ImportError(
                    "Annotated boxplots require the 'training' extra."
                ) from error
            annotation = Annotator(
                axis, [(0, 1)], data=table, x=label, y=descriptor, verbose=False
            )
            annotation.configure(
                test="Mann-Whitney",
                text_format="star",
                loc="outside",
                comparisons_correction="Benjamini-Hochberg",
            )
            annotation.apply_and_annotate()
    figure.tight_layout()
    return figure, axes
