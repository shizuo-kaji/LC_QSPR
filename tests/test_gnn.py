import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from lc_qspr.gnn import (  # noqa: E402
    GNNTrainingConfig,
    LiquidCrystalGraphDataset,
    predict_gnn_smiles,
    train_gnn_cross_validation,
)


def test_distributed_gnn_checkpoint_predicts():
    result = predict_gnn_smiles(
        ["CCCCCC1=CC=C(C=C1)C2=CC=CC=C2"],
        "models/GATv2Conv_phase_temp_cv0.pt",
        device="cpu",
    )

    assert result.loc[0, "predicted_phase_sequence"].startswith("Cr ")
    assert result.loc[0, "pred_Clearing"] == pytest.approx(
        float(result.loc[0, "pred_Clearing"])
    )


def test_joint_gnn_training_runs_one_epoch():
    smiles = ["CC", "CCC", "CCCC", "CCO", "CCN", "CCCl"]
    classification = pd.DataFrame({"Ntype": [0, 1, 0, 1, 0, 1]})
    regression = pd.DataFrame({"Melting": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    dataset = LiquidCrystalGraphDataset(
        smiles,
        classification_targets=classification,
        regression_targets=regression,
        atom_encoding="none",
    )
    result = train_gnn_cross_validation(
        dataset,
        [(np.array([2, 3, 4, 5]), np.array([0, 1]))],
        phase_targets=["Ntype"],
        temperature_targets=["Melting"],
        config=GNNTrainingConfig(
            hidden_channels=8,
            num_heads=2,
            num_layers=1,
            batch_size=2,
            epochs=1,
            warmup_epochs=1,
        ),
        device="cpu",
    )

    assert len(result.models) == 1
    assert result.predictions.index.tolist() == [0, 1]
    assert np.isfinite(result.histories[0]["validation"][0].loss)
