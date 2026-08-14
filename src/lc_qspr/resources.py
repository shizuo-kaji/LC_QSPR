"""Access package data without relying on the current working directory."""

from __future__ import annotations

from importlib.resources import files

import pandas as pd


def _resource(name: str):
    return files("lc_qspr.data").joinpath(name)


def descriptor_catalogue() -> pd.DataFrame:
    """Return the catalogue grouping Mordred descriptors by physical family."""
    with _resource("mordred_descriptors.csv").open("rb") as handle:
        return pd.read_csv(handle)


def model_descriptor_names() -> list[str]:
    """Return the ordered descriptor names expected by the bundled models."""
    with _resource("varnames.csv").open("rb") as handle:
        table = pd.read_csv(handle)
    if table.empty:
        raise ValueError("The packaged descriptor-name table is empty.")
    return table.iloc[:, -1].dropna().astype(str).tolist()

