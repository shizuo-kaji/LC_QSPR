"""Prepare LiqCryst-style phase records for modelling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import (
    ABSOLUTE_ZERO_C,
    CLEARING_LABELS,
    EXCLUDED_PHASE_LABELS,
    MELTING_LABELS,
    PHASE_TYPES,
    TEMPERATURE_TARGETS,
)
from .descriptors import calculate_descriptors


@dataclass(frozen=True)
class PhaseParseReport:
    """Counts produced while parsing raw phase strings."""

    rows: int
    excluded: int
    reentrant: int
    phase_labels: dict[str, int]


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV or Parquet data and use ``ID`` as a sorted index when present."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file was not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        table = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        table = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError("Supported table formats are CSV and Parquet.")
    if "ID" in table.columns:
        table = table.sort_values("ID").set_index("ID")
    return normalise_legacy_phase_columns(table)


def write_table(table: pd.DataFrame, path: str | Path) -> None:
    """Write a CSV or compressed Parquet table, preserving an ``ID`` index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = table.reset_index() if table.index.name == "ID" else table
    if path.suffix.lower() == ".csv":
        output.to_csv(path, index=False)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        output.to_parquet(path, index=False, compression="gzip")
    else:
        raise ValueError("Output must have a .csv, .parquet or .pq suffix.")


def normalise_legacy_phase_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Rename the historical ``D`` column prefix to the current ``Dh`` label."""
    rename = {}
    for suffix in ("type", "m", "p", "*type", "*m", "*p"):
        old, new = f"D{suffix}", f"Dh{suffix}"
        if old in table.columns and new not in table.columns:
            rename[old] = new
    return table.rename(columns=rename)


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _is_valid_temperature(value: str) -> bool:
    return (
        value != "?"
        and "<" not in value
        and ">" not in value
        and _is_number(value)
        and float(value) >= ABSOLUTE_ZERO_C
        and float(value) != 0
    )


def _looks_like_phase(value: str) -> bool:
    return not (
        _is_number(value)
        or "(" in value
        or "<" in value
        or ">" in value
        or "=0" in value
    )


def parse_phase_table(table: pd.DataFrame) -> tuple[pd.DataFrame, PhaseParseReport]:
    """Parse the LiqCryst ``Phases`` column into class and temperature columns."""
    required = {"SMILES", "Phases"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

    parsed = normalise_legacy_phase_columns(table.copy())
    parsed["Phases"] = (
        parsed["Phases"].astype(str)
        .str.replace(r"isSciDex.*", "is", regex=True)
        .str.replace(r"\s]\s", " J ", regex=True)
    )
    row_count = len(parsed)
    excluded = np.zeros(row_count, dtype=bool)
    reentrant = np.zeros(row_count, dtype=bool)
    phase_classes = {
        phase: np.zeros(row_count, dtype=np.int8)
        for phase in PHASE_TYPES + ("Clearing", "Melting")
    }
    temperature_columns = (
        tuple(f"{phase}m" for phase in PHASE_TYPES)
        + tuple(f"{phase}p" for phase in PHASE_TYPES)
        + ("Clearing", "Melting")
    )
    temperatures = {
        column: np.full(row_count, np.nan, dtype=np.float32)
        for column in temperature_columns
    }
    labels: dict[str, int] = {}

    for row_index, phase_string in enumerate(parsed["Phases"]):
        words = phase_string.split()
        for word_index, word in enumerate(words):
            if word in EXCLUDED_PHASE_LABELS:
                excluded[row_index] = True
            if _looks_like_phase(word) and not excluded[row_index]:
                labels[word] = labels.get(word, 0) + 1

            for phase in PHASE_TYPES:
                matched = word == phase or word == f"{phase}*"
                if not matched:
                    continue
                phase_classes[phase][row_index] = 2 if word.endswith("*") else 1
                neighbours = ((-1, f"{phase}m"), (1, f"{phase}p"))
                for offset, column in neighbours:
                    position = word_index + offset
                    if 0 <= position < len(words) and _is_valid_temperature(words[position]):
                        temperatures[column][row_index] = float(words[position])
                    else:
                        excluded[row_index] = True

            if word in MELTING_LABELS:
                phase_classes["Melting"][row_index] = 2 if word == "Tg" else 1
                if word_index + 1 < len(words) and _is_valid_temperature(words[word_index + 1]):
                    temperatures["Melting"][row_index] = float(words[word_index + 1])
            if word in CLEARING_LABELS:
                phase_classes["Clearing"][row_index] = 1 if word == "is" else 2
                if word_index > 0 and _is_valid_temperature(words[word_index - 1]):
                    temperatures["Clearing"][row_index] = float(words[word_index - 1])

        for phase in PHASE_TYPES:
            if phase_string.count(f" {phase} ") + phase_string.count(f" {phase}* ") > 1:
                reentrant[row_index] = True

    parsed["reentrant"] = reentrant
    parsed["exclude"] = excluded
    for phase in PHASE_TYPES:
        parsed[f"{phase}type"] = phase_classes[phase]
        parsed[f"{phase}m"] = temperatures[f"{phase}m"]
        parsed[f"{phase}p"] = temperatures[f"{phase}p"]
    for phase in ("Clearing", "Melting"):
        parsed[f"{phase}type"] = phase_classes[phase]
        parsed[phase] = temperatures[phase]

    report = PhaseParseReport(
        rows=row_count,
        excluded=int(excluded.sum()),
        reentrant=int(reentrant.sum()),
        phase_labels=dict(sorted(labels.items(), key=lambda item: item[1], reverse=True)),
    )
    return parsed, report


def separate_chiral_phases(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Convert class value 2 into independent starred-phase binary targets."""
    separated = normalise_legacy_phase_columns(table.copy())
    found = []
    for phase in PHASE_TYPES:
        type_column = f"{phase}type"
        if type_column not in separated:
            continue
        mask = separated[type_column] > 1
        if not mask.any():
            continue
        found.append(f"{phase}*")
        separated[f"{phase}*type"] = 0
        separated.loc[mask, f"{phase}*type"] = 1
        separated.loc[mask, type_column] = 0
        for suffix in ("p", "m"):
            source, destination = f"{phase}{suffix}", f"{phase}*{suffix}"
            separated.loc[mask, destination] = separated.loc[mask, source]
            separated.loc[mask, source] = np.nan
    return separated, found


def filter_training_table(
    table: pd.DataFrame,
    *,
    remove_duplicate_smiles: bool = True,
    minimum_nitrogen_and_carbon: int = 12,
    suspicious_tokens: Iterable[str] = ("?", " 0 ", "X", "S", "no", "un", "ex"),
) -> pd.DataFrame:
    """Apply the quality filters used by the original training workflow."""
    filtered = normalise_legacy_phase_columns(table.copy())
    if remove_duplicate_smiles:
        counts = filtered["SMILES"].value_counts()
        filtered = filtered[~filtered["SMILES"].isin(counts[counts > 1].index)]
    if minimum_nitrogen_and_carbon > 0:
        required = {"nC", "nN"}
        if not required.issubset(filtered.columns):
            raise ValueError("nC and nN descriptors are required for the size filter.")
        filtered = filtered[
            (filtered["nC"] + filtered["nN"]) >= minimum_nitrogen_and_carbon
        ]
    for flag in ("exclude", "prohibited", "reentrant"):
        if flag in filtered:
            filtered = filtered[~filtered[flag].fillna(False).astype(bool)]
    if "Clearingtype" in filtered:
        filtered = filtered[filtered["Clearingtype"] == 1]
    for phase in PHASE_TYPES:
        lower, upper = f"{phase}m", f"{phase}p"
        if lower in filtered and upper in filtered:
            filtered = filtered[~(filtered[lower] > filtered[upper])]
    for target in TEMPERATURE_TARGETS:
        if target in filtered:
            invalid = (
                (filtered[target] < ABSOLUTE_ZERO_C)
                | (filtered[target] >= 800)
                | (filtered[target] == 0)
            )
            filtered.loc[invalid, target] = np.nan
    if "Phases" in filtered:
        for token in suspicious_tokens:
            filtered = filtered[
                ~filtered["Phases"].str.contains(token, regex=False, na=False)
            ]
    available_temperatures = [c for c in TEMPERATURE_TARGETS if c in filtered]
    if available_temperatures:
        filtered = filtered[filtered[available_temperatures].notna().any(axis=1)]
    return filtered.copy()


def add_mordred_descriptors(
    table: pd.DataFrame, *, n_jobs: int = 1, show_progress: bool = False
) -> pd.DataFrame:
    """Calculate Mordred descriptors and append them without changing row order."""
    if "SMILES" not in table:
        raise ValueError("Input table must contain a SMILES column.")
    descriptors = calculate_descriptors(
        table["SMILES"].tolist(), n_jobs=n_jobs, show_progress=show_progress
    )
    descriptors.index = table.index
    overlap = [column for column in descriptors if column in table]
    return pd.concat([table.drop(columns=overlap), descriptors], axis=1)
