"""High-level CatBoost inference API."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from .descriptors import calculate_descriptors
from .phases import CHIRAL_PHASE_TYPES, TEMPERATURE_TARGETS, format_phase_sequence
from .resources import model_descriptor_names


class CatBoostPredictor:
    """Predict liquid-crystal phases and transition temperatures from SMILES."""

    def __init__(
        self,
        phase_model: str | Path,
        temperature_model: str | Path,
        *,
        threshold: float = 0.5,
    ) -> None:
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1.")
        phase_model = Path(phase_model)
        temperature_model = Path(temperature_model)
        if not phase_model.is_file():
            raise FileNotFoundError(f"Phase model was not found: {phase_model}")
        if not temperature_model.is_file():
            raise FileNotFoundError(f"Temperature model was not found: {temperature_model}")
        self.threshold = threshold
        self.phase_model = CatBoostClassifier()
        self.temperature_model = CatBoostRegressor()
        self.phase_model.load_model(str(phase_model))
        self.temperature_model.load_model(str(temperature_model))
        self.descriptor_names = model_descriptor_names()

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        phase_filename: str = "CatBoost_phase.cbm",
        temperature_filename: str = "CatBoost_temperature_7000epochs.cbm",
        threshold: float = 0.5,
    ) -> "CatBoostPredictor":
        """Load a pair of CatBoost models from ``directory``.

        Custom filenames make this suitable for models produced by the training
        API as well as for the models distributed with LC-QSPR.
        """
        directory = Path(directory)
        return cls(
            directory / phase_filename,
            directory / temperature_filename,
            threshold=threshold,
        )

    def predict(self, smiles: Sequence[str]) -> pd.DataFrame:
        """Return phase probabilities, decisions, and temperatures."""
        smiles = list(smiles)
        descriptors = calculate_descriptors(smiles)
        missing = sorted(set(self.descriptor_names) - set(descriptors.columns))
        if missing:
            preview = ", ".join(missing[:5])
            raise RuntimeError(f"Required Mordred descriptors are unavailable: {preview}")

        features = descriptors.loc[:, self.descriptor_names]
        phase_probabilities = np.asarray(self.phase_model.predict_proba(features))
        temperatures = np.asarray(self.temperature_model.predict(features))
        if phase_probabilities.shape[1] != len(CHIRAL_PHASE_TYPES):
            raise RuntimeError("The phase model output does not match the package vocabulary.")
        if temperatures.shape[1] != len(TEMPERATURE_TARGETS):
            raise RuntimeError("The temperature model output does not match the package vocabulary.")

        result = pd.DataFrame({"SMILES": smiles})
        for index, target in enumerate(CHIRAL_PHASE_TYPES):
            result[f"pred_{target}type"] = phase_probabilities[:, index] >= self.threshold
            result[f"prob_{target}type"] = phase_probabilities[:, index]
        for index, target in enumerate(TEMPERATURE_TARGETS):
            result[f"pred_{target}"] = temperatures[:, index]
        result.insert(
            1,
            "predicted_phase_sequence",
            [format_phase_sequence(row) for row in result.to_dict(orient="records")],
        )
        result.index.name = "ID"
        return result


def predict_smiles(
    smiles: Sequence[str],
    model_directory: str | Path = "models",
    *,
    phase_model: str | Path | None = None,
    temperature_model: str | Path | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Predict with the distributed or an explicitly selected CatBoost pair.

    ``model_directory`` uses the standard distributed filenames. To use models
    produced by a user's own training run, supply both ``phase_model`` and
    ``temperature_model``. Requiring a complete pair avoids accidentally mixing
    models from different training runs.
    """
    supplied_models = (phase_model is not None, temperature_model is not None)
    if any(supplied_models) and not all(supplied_models):
        raise ValueError(
            "phase_model and temperature_model must be supplied together."
        )
    if all(supplied_models):
        predictor = CatBoostPredictor(
            phase_model,
            temperature_model,
            threshold=threshold,
        )
    else:
        predictor = CatBoostPredictor.from_directory(
            model_directory,
            threshold=threshold,
        )
    return predictor.predict(smiles)
