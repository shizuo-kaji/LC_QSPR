"""Mordred descriptor calculation from SMILES strings."""

from __future__ import annotations

from collections.abc import Sequence
import os
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem

from .resources import descriptor_catalogue


def _load_mordred():
    # Mordred 1.2 imports ``numpy.product``, removed in NumPy 2.  Keeping this
    # local alias lets existing scientific environments remain usable.
    if not hasattr(np, "product"):
        np.product = np.prod  # type: ignore[attr-defined]
    from mordred import Calculator, descriptors

    return Calculator, descriptors


def calculate_descriptors(
    smiles: Sequence[str], *, n_jobs: int = 1, show_progress: bool = False
) -> pd.DataFrame:
    """Calculate 2D Mordred descriptors.

    Parameters
    ----------
    smiles:
        One or more SMILES strings. Invalid strings are reported together so
        that they can be corrected in the input table.

    Returns
    -------
    pandas.DataFrame
        One row per input molecule. Calculation failures for individual
        descriptors are represented by ``NaN``, as expected by CatBoost.
    """
    smiles = list(smiles)
    if not smiles:
        raise ValueError("At least one SMILES string is required.")

    molecules = [Chem.MolFromSmiles(value) if isinstance(value, str) else None for value in smiles]
    invalid = [index for index, molecule in enumerate(molecules) if molecule is None]
    if invalid:
        locations = ", ".join(str(index) for index in invalid)
        raise ValueError(f"Invalid SMILES at row(s): {locations}")

    Calculator, descriptors = _load_mordred()
    calculator = Calculator(descriptors, ignore_3D=True)
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be -1 or a positive integer.")
    if n_jobs == -1:
        n_jobs = max((os.cpu_count() or 2) - 1, 1)
    records = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = calculator.map(
            molecules,
            nproc=n_jobs,
            nmols=len(molecules),
            quiet=not show_progress,
            ipynb=show_progress,
        )
        for result in results:
            result = result.fill_missing(np.nan)
            records.append({str(name): value for name, value in result.asdict().items()})
    return pd.DataFrame.from_records(records)


def descriptor_families() -> dict[str, tuple[str, ...]]:
    """Return the descriptor families and ablation sets used in the paper."""
    Calculator, descriptors = _load_mordred()
    catalogue = descriptor_catalogue()
    families: dict[str, tuple[str, ...]] = {}
    for family in ("Shape", "Electric", "Flexible"):
        descriptor_modules = catalogue.loc[catalogue[family] == 1, "name"]
        calculator = Calculator(
            [getattr(descriptors, module) for module in descriptor_modules]
        )
        families[family] = tuple(str(value) for value in calculator.descriptors)
    all_descriptors = tuple(
        str(value)
        for value in Calculator(descriptors, ignore_3D=True).descriptors
    )
    families["All"] = all_descriptors
    for name, components in {
        "ShapeElectric": ("Shape", "Electric"),
        "ShapeFlexible": ("Shape", "Flexible"),
        "ElectricFlexible": ("Electric", "Flexible"),
    }.items():
        families[name] = tuple(
            dict.fromkeys(
                descriptor
                for component in components
                for descriptor in families[component]
            )
        )
    all_set = set(all_descriptors)
    for family in ("Flexible", "Shape", "Electric"):
        families[f"-{family}"] = tuple(
            descriptor
            for descriptor in all_descriptors
            if descriptor in all_set - set(families[family])
        )
    return families


def select_descriptor_names(
    table: pd.DataFrame, family: str = "All"
) -> list[str]:
    """Select available descriptor columns for a named physical family."""
    families = descriptor_families()
    if family not in families:
        choices = ", ".join(families)
        raise ValueError(f"Unknown descriptor family {family!r}. Choose from: {choices}")
    names = [name for name in families[family] if name in table.columns]
    if not names:
        raise ValueError(f"No {family!r} Mordred descriptors are present in the table.")
    return names
