from pathlib import Path

import pandas as pd
import pytest

from lc_qspr import cli


def test_cli_passes_explicit_catboost_models(monkeypatch, tmp_path):
    captured = {}

    def fake_predict(smiles, model_directory, **kwargs):
        captured.update(smiles=smiles, model_directory=model_directory, **kwargs)
        return pd.DataFrame(
            {"SMILES": smiles, "predicted_phase_sequence": ["Cr 100.00 is"]}
        )

    output = tmp_path / "prediction.csv"
    monkeypatch.setattr(cli, "predict_smiles", fake_predict)
    monkeypatch.setattr(
        "sys.argv",
        [
            "lcqspr",
            "--smiles",
            "CC",
            "--phase-model",
            "run/phase.cbm",
            "--temperature-model",
            "run/temperature.cbm",
            "--output",
            str(output),
        ],
    )

    cli.main()

    assert captured["phase_model"] == Path("run/phase.cbm")
    assert captured["temperature_model"] == Path("run/temperature.cbm")
    assert output.is_file()


def test_cli_rejects_an_incomplete_catboost_pair(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["lcqspr", "--smiles", "CC", "--phase-model", "run/phase.cbm"],
    )

    with pytest.raises(SystemExit, match="must be supplied together"):
        cli.main()


def test_cli_passes_an_explicit_gnn_checkpoint(monkeypatch, tmp_path):
    gnn = pytest.importorskip("lc_qspr.gnn")
    captured = {}

    def fake_predict(smiles, checkpoint, **kwargs):
        captured.update(smiles=smiles, checkpoint=checkpoint, **kwargs)
        return pd.DataFrame(
            {"SMILES": smiles, "predicted_phase_sequence": ["Cr 100.00 is"]}
        )

    output = tmp_path / "prediction.csv"
    monkeypatch.setattr(gnn, "predict_gnn_smiles", fake_predict)
    monkeypatch.setattr(
        "sys.argv",
        [
            "lcqspr",
            "--smiles",
            "CC",
            "--gnn-model",
            "run/model.pt",
            "--output",
            str(output),
        ],
    )

    cli.main()

    assert captured["checkpoint"] == Path("run/model.pt")
    assert output.is_file()
