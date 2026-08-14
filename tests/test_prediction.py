from pathlib import Path

import pytest

from lc_qspr import CatBoostPredictor, predict_smiles


def test_pretrained_models_predict_one_molecule():
    predictor = CatBoostPredictor.from_directory(Path("models"))
    result = predictor.predict(["CCCCCC1=CC=C(C=C1)C2=CC=CC=C2"])

    assert len(result) == 1
    assert result.loc[0, "SMILES"].startswith("CCCCCC")
    assert result.loc[0, "predicted_phase_sequence"].startswith("Cr ")
    assert result.loc[0, "predicted_phase_sequence"].endswith(" is")
    assert result[["pred_Melting", "pred_Clearing"]].notna().all().all()


def test_explicit_catboost_model_files_predict_one_molecule():
    result = predict_smiles(
        ["CCCCCC1=CC=C(C=C1)C2=CC=CC=C2"],
        phase_model=Path("models/CatBoost_phase.cbm"),
        temperature_model=Path("models/CatBoost_temperature_7000epochs.cbm"),
    )

    assert len(result) == 1
    assert result.loc[0, "predicted_phase_sequence"].startswith("Cr ")


def test_explicit_catboost_models_require_a_complete_pair():
    with pytest.raises(ValueError, match="must be supplied together"):
        predict_smiles(
            ["CC"],
            phase_model=Path("models/CatBoost_phase.cbm"),
        )


def test_catboost_directory_accepts_custom_filenames():
    predictor = CatBoostPredictor.from_directory(
        Path("models"),
        phase_filename="CatBoost_phase.cbm",
        temperature_filename="CatBoost_temperature_7000epochs.cbm",
    )

    result = predictor.predict(["CCCCCC1=CC=C(C=C1)C2=CC=CC=C2"])
    assert len(result) == 1
