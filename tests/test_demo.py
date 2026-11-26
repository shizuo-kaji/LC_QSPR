from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

from lc_qspr.batch import SUMMARY_COLUMNS, predict_smiles_list, read_smiles_list
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


def test_batch_inputs_and_alignment():
    raw = 'ID,smiles\r\n001,OCC\r\n"two, β",C1CC\r\n003,CCO\r\n004,\r\n'
    inputs = read_smiles_list("list.csv", raw.encode("utf-8-sig"))
    assert inputs["ID"].tolist() == ["001", "two, β", "003", "004"]
    assert read_smiles_list("list.txt", b"CCO\n\nCC\r\n")["SMILES"].tolist() == ["CCO", "CC"]
    assert read_smiles_list("list.csv", b"CCO\nCC\n")["SMILES"].tolist() == ["CCO", "CC"]
    for invalid in (b"", b"SMILES\n", b"name,value\nx,y\n"):
        with pytest.raises(ValueError):
            read_smiles_list("list.csv", invalid)

    def predict(smiles):
        assert smiles == ["CCO"]
        return pd.DataFrame(
            [["CCO", "Cr 10.00 is", 10.0, 20.0]], columns=["SMILES", *SUMMARY_COLUMNS]
        )

    result = predict_smiles_list(inputs, predict, "CatBoost")
    assert result["SMILES"].tolist() == ["OCC", "C1CC", "CCO", ""]
    assert result["ID"].tolist() == inputs["ID"].tolist()
    assert result["error"].ne("").tolist() == [False, True, False, True]
    assert result.loc[[0, 2], "pred_Melting"].tolist() == [10.0, 10.0]
    assert result.loc[[1, 3], SUMMARY_COLUMNS].isna().all().all()
    invalid_only = predict_smiles_list(inputs.iloc[[1, 3]], predict, "CatBoost")
    assert invalid_only["error"].ne("").all()
    assert invalid_only[SUMMARY_COLUMNS].isna().all().all()


def test_smi_preserves_names_and_extended_smiles():
    contents = "CCO\t001\nCC\nOCC Éthanol sample\nCC |$C1;C2$| atoms 1 and 2\n"
    inputs = read_smiles_list("named.smi", contents.encode())
    assert inputs["SMILES"].tolist() == ["CCO", "CC", "OCC", "CC |$C1;C2$|"]
    assert inputs["ID"].tolist() == ["001", "", "Éthanol sample", "atoms 1 and 2"]
    assert read_smiles_list("unnamed.smi", b"CCO\nCC\n").columns.tolist() == ["SMILES"]


def test_batch_upload_model_switch_and_download(monkeypatch):
    pytest.importorskip("streamlit_ketcher_editor")
    pytest.importorskip("torch_geometric")
    from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
    from streamlit.testing.v1 import AppTest

    from lc_qspr.gnn import predict_gnn_smiles
    from lc_qspr.prediction import predict_smiles

    media = MemoryMediaFileStorage("/mock/media")
    monkeypatch.setattr("streamlit.testing.v1.app_test.MemoryMediaFileStorage", lambda endpoint: media)
    app_path = Path(__file__).resolve().parents[1] / "demo" / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=60).run()
    uploaded = ("list.csv", b"ID,SMILES\n001,CCO\n002,C1CC\n003,OCC\n", "text/csv")
    app.file_uploader[0].set_value(uploaded).run()
    for model, expected in (
        ("GATv2 GNN", predict_gnn_smiles(["CCO"], "models/GATv2Conv_phase_temp_cv0.pt", device="cpu")),
        ("CatBoost", predict_smiles(["CCO"])),
    ):
        app.get("button_group")[0].set_value(model).run()
        assert not app.get("download_button")
        next(button for button in app.button if button.label == "Predict list").click().run()
        assert not app.exception and not app.error
        result = app.dataframe[0].value
        assert result["model"].tolist() == [model] * 3
        assert result["ID"].tolist() == ["001", "002", "003"]
        assert result["error"].ne("").tolist() == [False, True, False]
        assert result.loc[[0, 2], "pred_Clearing"].tolist() == pytest.approx([expected.at[0, "pred_Clearing"]] * 2)
        download = app.get("download_button")[0]
        assert download.label == "Download results (CSV)"
        saved = media.get_file(Path(download.proto.url).name)
        assert saved.mimetype == "text/csv"
        assert saved.filename == f"list-{model.split()[0].lower()}-predictions.csv"
        exported = pd.read_csv(
            BytesIO(saved.content), dtype={"ID": str}, keep_default_na=False,
            na_values={column: [""] for column in SUMMARY_COLUMNS},
        )
        pd.testing.assert_frame_equal(exported, result, check_dtype=False)
    app.file_uploader[0].set_value(("list.csv", b"SMILES\nCC\n", "text/csv")).run()
    assert not app.get("download_button")
    assert not app.dataframe
