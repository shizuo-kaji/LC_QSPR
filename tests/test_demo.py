from pathlib import Path

import pytest
from rdkit import Chem

from lc_qspr.structures import molecule_from_smiles


def test_sketch_input_keeps_stereochemistry_and_rejects_incomplete_structures():
    smiles = "C[C@H](O)F"
    assert Chem.MolToSmiles(molecule_from_smiles(smiles)) == smiles
    for invalid in ("", "C1CC", "CCO trailing text", "*C"):
        with pytest.raises(ValueError):
            molecule_from_smiles(invalid)


def test_editor_reference_and_model_switch_use_matching_predictions():
    pytest.importorskip("streamlit_ketcher_editor")
    pytest.importorskip("torch_geometric")
    from streamlit.testing.v1 import AppTest

    from lc_qspr.gnn import predict_gnn_smiles
    from lc_qspr.prediction import predict_smiles

    app_path = Path(__file__).resolve().parents[1] / "demo" / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=60).run()

    def click(label):
        next(button for button in app.button if button.label == label).click().run()
        assert not app.exception

    app.text_input[0].set_value("OCC")
    click("Load reference")
    assert app.session_state["reference"] == "CCO"
    click("6CB")
    edited = app.session_state["editor_seed"]
    molecules = ["CCO", edited]
    for model, expected in (
        ("GATv2 GNN", predict_gnn_smiles(molecules, "models/GATv2Conv_phase_temp_cv0.pt", device="cpu")),
        ("CatBoost", predict_smiles(molecules)),
    ):
        app.get("button_group")[0].set_value(model).run()
        assert not app.error
        assert app.session_state["editor_seed"] == edited
        for target, metric in zip(("Melting", "Clearing"), app.metric, strict=True):
            original, modified = expected[f"pred_{target}"]
            assert metric.value == f"{modified:.1f} °C"
            assert metric.delta == f"{modified - original:+.1f} °C"

    click("Use edited structure as reference")
    assert app.session_state["reference"] == edited
    assert all(not metric.delta for metric in app.metric)
    click("8CB")
    click("Restore reference")
    assert app.session_state["editor_seed"] == edited
    app.text_input[0].set_value("C1(")
    click("Load reference")
    assert app.error
    assert app.session_state["reference"] == edited
