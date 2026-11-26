"""Read SMILES lists and align batch predictions with the original rows."""

from __future__ import annotations

from collections.abc import Callable
from io import StringIO
from pathlib import Path

import pandas as pd
from rdkit import Chem

from .structures import molecule_from_smiles


SUMMARY_COLUMNS = ["predicted_phase_sequence", "pred_Melting", "pred_Clearing"]


def read_smiles_list(filename: str, contents: bytes) -> pd.DataFrame:
    """Read a UTF-8 CSV or a plain-text list without coercing identifiers to numbers."""
    text = contents.decode("utf-8-sig")
    suffix = Path(filename).suffix.lower()
    if not text.strip():
        raise ValueError("The file is empty.")

    if suffix == ".csv":
        table = pd.read_csv(StringIO(text), header=None, dtype=str, keep_default_na=False)
        headers = table.iloc[0].str.strip().str.casefold().tolist()
        if headers.count("smiles") == 1:
            result = pd.DataFrame({"SMILES": table.iloc[1:, headers.index("smiles")]})
            if "id" in headers:
                result.insert(0, "ID", table.iloc[1:, headers.index("id")])
        elif table.shape[1] == 1:
            result = table.set_axis(["SMILES"], axis="columns")
        else:
            raise ValueError("A CSV with multiple columns needs one column named SMILES.")
    elif suffix in {".smi", ".txt"}:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        result = pd.DataFrame({"SMILES": lines})
        if suffix == ".smi":
            # In named SMILES files, a CXSMILES block belongs to the structure.
            fields = result["SMILES"].str.extract(r"^(\S+(?:\s+\|[^|]*\|)?)(?:\s+(.+))?$")
            result["SMILES"] = fields[0]
            if fields[1].notna().any():
                result.insert(0, "ID", fields[1].fillna(""))
    else:
        raise ValueError("Choose a CSV, SMI or TXT file.")

    if result.empty:
        raise ValueError("The file contains no molecules.")
    return result.reset_index(drop=True)


def predict_smiles_list(
    inputs: pd.DataFrame, predict: Callable[[list[str]], pd.DataFrame], model: str
) -> pd.DataFrame:
    """Predict unique molecules in batches and retain duplicates and invalid input rows."""
    canonical = []
    errors = []
    for smiles in inputs["SMILES"]:
        try:
            canonical.append(Chem.MolToSmiles(molecule_from_smiles(smiles)))
            errors.append("")
        except ValueError as error:
            canonical.append(None)
            errors.append(str(error))

    unique = list(dict.fromkeys(value for value in canonical if value is not None))
    # Mordred constructs a wide descriptor table. Bound its peak size on the shared CPU.
    batch_size = 64
    summaries = [
        predict(unique[start:start + batch_size]).set_index("SMILES")[SUMMARY_COLUMNS]
        for start in range(0, len(unique), batch_size)
    ]
    lookup = pd.concat(summaries) if summaries else pd.DataFrame(columns=SUMMARY_COLUMNS)
    aligned = lookup.reindex(canonical).reset_index(drop=True)
    result = pd.concat([inputs.reset_index(drop=True), aligned], axis="columns")
    result["model"] = model
    result["error"] = errors
    return result
