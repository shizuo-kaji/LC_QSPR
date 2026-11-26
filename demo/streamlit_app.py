"""Interactive liquid-crystal structure editing and phase prediction."""

import os
from functools import partial
from html import escape
from pathlib import Path

import streamlit as st
from rdkit import Chem
from streamlit_ketcher_editor import st_ketcher

from lc_qspr.batch import predict_smiles_list, read_smiles_list
from lc_qspr.prediction import CatBoostPredictor
from lc_qspr.structures import molecule_from_smiles, structure_svg


APP_DIR = Path(__file__).resolve().parent
SOURCE_DIR = APP_DIR.parent / "src"
# Streamlit watches imported modules outside the entry point's directory via PYTHONPATH.
python_path = os.environ.get("PYTHONPATH", "")
if str(SOURCE_DIR) not in python_path.split(os.pathsep):
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (str(SOURCE_DIR), python_path)))

MODEL_DIR = Path(os.environ.get("LC_QSPR_MODEL_DIR", APP_DIR.parent / "models"))
MODELS = {
    "GATv2 GNN": (
        Path(os.environ.get(
            "LC_QSPR_GNN_MODEL", MODEL_DIR / "GATv2Conv_phase_temp_cv0.pt"
        )),
    ),
    "CatBoost": (
        Path(os.environ.get("LC_QSPR_PHASE_MODEL", MODEL_DIR / "CatBoost_phase.cbm")),
        Path(os.environ.get(
            "LC_QSPR_TEMPERATURE_MODEL", MODEL_DIR / "CatBoost_temperature_7000epochs.cbm"
        )),
    ),
}
HOMOLOGUES = {
    f"{length}CB": "C" * length + "c1ccc(-c2ccc(C#N)cc2)cc1" for length in range(5, 9)
}


def prediction_revision(paths):
    # Streamlit's function cache does not track imported code, package data or weights.
    package = SOURCE_DIR / "lc_qspr"
    files = [
        Path(__file__), *sorted(package.rglob("*.py")),
        *sorted((package / "data").glob("*.csv")), *paths,
    ]
    return tuple((str(path), path.stat().st_mtime_ns if path.is_file() else None) for path in files)


@st.cache_resource(show_spinner=False, max_entries=2)
def catboost_models(phase_path, temperature_path, revision):
    return CatBoostPredictor(phase_path, temperature_path)


def predict_molecules(smiles, model, paths, revision):
    if model == "GATv2 GNN":
        from lc_qspr.gnn import predict_gnn_smiles

        return predict_gnn_smiles(smiles, paths[0], device="cpu")
    return catboost_models(*paths, revision).predict(smiles)


@st.cache_data(scope="session", max_entries=64, show_spinner=False)
def prediction(smiles, model, paths, revision):
    row = predict_molecules([smiles], model, paths, revision).iloc[0]
    return {
        "sequence": row["predicted_phase_sequence"],
        "Melting": float(row["pred_Melting"]),
        "Clearing": float(row["pred_Clearing"]),
    }


def open_sketch(smiles):
    # Ketcher reads its initial molecule on mount. Preserve its undo history
    # on model switches and replace the editor only when loading a structure.
    st.session_state.editor_seed = smiles
    st.session_state.editor_revision += 1


def pin_reference(smiles):
    st.session_state.reference = smiles
    st.session_state.reference_input = smiles
    st.session_state.input_error = ""


def load_reference():
    try:
        molecule = molecule_from_smiles(st.session_state.reference_input)
    except ValueError as error:
        st.session_state.input_error = str(error)
        return
    smiles = Chem.MolToSmiles(molecule)
    pin_reference(smiles)
    open_sketch(smiles)


def show_structure(label, molecule, summary):
    st.markdown(f"**{label}**")
    st.image(structure_svg(molecule), width="stretch")
    parts = []
    for token in summary["sequence"].split():
        if token[0].isalpha():
            parts.append(f'<b class="phase">{escape(token)}</b>')
        else:
            parts.append(f'<span>{escape(token)}</span>')
    st.html('<div class="phase-sequence">' + " ".join(parts) + "</div>")


def show_predictions(model, edited, revision):
    reference = st.session_state.reference
    current = Chem.MolToSmiles(edited) if edited is not None else None
    with st.spinner("Predicting transitions…"):
        try:
            original = prediction(reference, model, MODELS[model], revision)
            modified = prediction(current, model, MODELS[model], revision) if current else None
        except FileNotFoundError as error:
            st.error(f"Could not load the {model} weights. Check the configured model paths. {error}")
            return
        except (ImportError, RuntimeError, ValueError, OSError) as error:
            st.error(f"{model} prediction could not be completed. {error}")
            return

    if modified is not None:
        with st.container(border=True, key="edited_prediction"):
            show_structure("Edited structure", edited, modified)
            for column, target in zip(st.columns(2), ("Melting", "Clearing")):
                delta = None
                if current != reference:
                    delta = f"{modified[target] - original[target]:+.1f} °C"
                column.metric(target, f"{modified[target]:.1f} °C", delta, delta_color="off")
            if current != reference:
                st.caption("Temperature changes are relative to the reference.")

    with st.container(border=True, key="reference_prediction"):
        show_structure("Reference", molecule_from_smiles(reference), original)
    st.caption(
        "Temperatures in °C. Each mesophase sits between its predicted lower and upper "
        "temperatures. Cr = crystal, N = nematic, is = isotropic liquid."
    )


def show_editor(model, revision):
    state = st.session_state
    if "reference" not in state:
        pin_reference(HOMOLOGUES["5CB"])
        state.editor_seed = state.reference
        state.editor_revision = 0

    with st.form("smiles_form", border=False):
        field, submit = st.columns([4, 1], vertical_alignment="bottom")
        field.text_input("Reference SMILES", key="reference_input")
        submit.form_submit_button("Load reference", on_click=load_reference, width="stretch")
    if state.input_error:
        st.error(state.input_error)

    sketch, results = st.columns([1.4, 1], gap="large")
    with sketch:
        st.subheader("Molecule editor")
        st.write(
            "Choose an element and click an atom to replace it. Drag a bond from a chain end "
            "to extend it. Predictions update as you edit."
        )
        with st.container(horizontal=True, vertical_alignment="center", key="homologues"):
            st.markdown(
                "**Try a homologue**", width="content",
                help="Each cyanobiphenyl in this series adds one CH₂ to the alkyl chain.",
            )
            for name, smiles in HOMOLOGUES.items():
                st.button(name, on_click=open_sketch, args=(smiles,))
        raw = st_ketcher(state.editor_seed, height=385, key=f"sketch_{state.editor_revision}")
        try:
            edited = molecule_from_smiles(raw or "")
        except ValueError as error:
            edited = None
            st.warning(str(error))

        with st.container(horizontal=True):
            st.button("Restore reference", on_click=open_sketch, args=(state.reference,))
            current = Chem.MolToSmiles(edited) if edited is not None else None
            st.button(
                "Use edited structure as reference", on_click=pin_reference, args=(current,),
                disabled=current is None or current == state.reference,
            )
        if current:
            with st.expander("Edited SMILES"):
                st.code(current, language=None, wrap_lines=True)

    with results:
        st.subheader("Predicted transitions")
        show_predictions(model, edited, revision)


def show_batch(model, revision):
    st.write("Use a UTF-8 CSV with a SMILES column and an optional ID, or a text file with one SMILES per line.")
    upload = st.file_uploader(
        "Upload a SMILES list", type=["csv", "smi", "txt"], max_upload_size=2,
    )
    state = st.session_state
    if upload is None:
        state.pop("batch_result", None)
        return

    request = (upload.file_id, model, revision)
    saved_request, table = state.get("batch_result", (None, None))
    if saved_request != request:
        state.pop("batch_result", None)
        table = None
    if st.button("Predict list", type="primary"):
        state.pop("batch_result", None)
        try:
            inputs = read_smiles_list(upload.name, upload.getvalue())
            predict = partial(predict_molecules, model=model, paths=MODELS[model], revision=revision)
            with st.spinner(f"Predicting {len(inputs)} rows with {model}…"):
                table = predict_smiles_list(inputs, predict, model)
        except UnicodeDecodeError:
            st.error("The file could not be read as UTF-8. Save it as UTF-8 and upload it again.")
            return
        except (ValueError, ImportError, RuntimeError, OSError) as error:
            st.error(str(error))
            return
        state.batch_result = (request, table)

    if table is not None:
        count = table["error"].eq("").sum()
        st.caption(f"{model} · {count} of {len(table)} rows predicted")
        st.dataframe(table, hide_index=True, column_config={
            "model": None,
            "predicted_phase_sequence": "Predicted phase sequence",
            "pred_Melting": st.column_config.NumberColumn("Melting (°C)", format="%.2f"),
            "pred_Clearing": st.column_config.NumberColumn("Clearing (°C)", format="%.2f"),
            "error": "Input error",
        })
        st.download_button(
            "Download results (CSV)", table.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{Path(upload.name).stem}-{model.split()[0].lower()}-predictions.csv",
            mime="text/csv", on_click="ignore",
        )


def main():
    st.set_page_config(page_title="LC-QSPR phase explorer", layout="wide")
    st.html(APP_DIR / "style.css")
    title, switch, repository = st.columns([2.3, 1.1, 0.8], vertical_alignment="bottom")
    title.title("LC-QSPR phase explorer")
    model = switch.segmented_control(
        "Prediction model", list(MODELS), default="GATv2 GNN", required=True, key="model"
    )
    repository.link_button(
        "GitHub repository", "https://github.com/shizuo-kaji/LC_QSPR", icon=":material/open_in_new:"
    )
    revision = prediction_revision(MODELS[model])
    editor, batch = st.tabs(["Single molecule", "Batch predictions"])
    with editor:
        show_editor(model, revision)
    with batch:
        show_batch(model, revision)


if __name__ == "__main__":
    main()
