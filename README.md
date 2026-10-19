# LC-QSPR

**LC-QSPR** predicts liquid-crystal mesophases and transition temperatures (°C) directly from molecular structures represented as SMILES strings.

---

## What LC-QSPR Predicts

Given one or more SMILES strings (readily exported from ChemDraw, PubChem or RDKit), LC-QSPR outputs:

1. **Readable phase sequence (`predicted_phase_sequence`)** in standard LiqCryst notation, ordered by transition temperature in °C:
   ```text
   Cr 76.8 A 112.5 N* 119.8 is
   ```
2. **Phase existence and probabilities** across **17 liquid-crystal phase targets**:
   - **Achiral phases (12):** `N` (nematic), `A` (smectic A), `B`, `C`, `E`, `F`, `G`, `H`, `I`, `J`, `Dh` (hexagonal discotic) and `BP` (blue phase).
   - **Chiral phases (5):** `N*`, `C*`, `E*`, `F*` and `I*`.
3. **Transition temperatures (°C)** across **26 temperature targets**:
   - `Melting` (`Cr` → mesophase or isotropic liquid) and `Clearing` (mesophase → isotropic liquid `is`).
   - Lower (`<Phase>m`) and upper (`<Phase>p`) temperature bounds for each mesophase (for example, `Nm` and `Np` for the nematic range).

Two pre-trained model families are bundled in `models/` and ready to use out of the box:

| Model family | Bundled files | Input representation | Best suited for |
|---|---|---|---|
| **CatBoost** | `CatBoost_phase.cbm`<br>`CatBoost_temperature_7000epochs.cbm` | 1,613 2D Mordred descriptors | Fast screening on CPU, tabular feature importance (SHAP) and physical descriptor-family analysis (`Shape`, `Electric`, `Flexible`) |
| **GNN (GATv2Conv)** | `GATv2Conv_phase_temp_cv0.pt` | Molecular graph (atom & bond attributes) | End-to-end joint phase and temperature prediction directly from molecular topology without descriptor calculation |

---

## Installation

Python 3.10 or later is required. Run the command matching your workflow from the repository root:

```bash
# 1. Quick prediction with the bundled CatBoost models (CPU only, minimal dependencies)
python -m pip install -e .

# 2. Interactive Jupyter notebooks and tree-model (CatBoost / LightGBM) training
python -m pip install -e ".[notebook,training]"

# 3. Complete workflow, including PyTorch Geometric for GNN inference and training
python -m pip install -e ".[notebook,training,gnn]"
```

No GPU is required for inference with either CatBoost or the GNN. Full 5-fold GNN training is substantially faster on a CUDA or Apple Silicon (MPS) accelerator.

---

## Quick Start for Chemistry Researchers

### Option A — Interactive Jupyter Notebook (Recommended)

Launch JupyterLab and open [00_predict.ipynb](notebooks/00_predict.ipynb):

```bash
jupyter lab notebooks/00_predict.ipynb
```

This notebook draws the input molecules, runs both the **CatBoost** and **GNN** models on the same SMILES list, compares the predicted phase sequences against measured literature values, and exports the full prediction tables to `results/`.

### Option B — Command-Line / Spreadsheet Workflow (No Python coding required)

1. Export or save your candidate structures as a CSV file with a `SMILES` column (see `examples/data/sample.csv` for an example).
2. Run `lcqspr` from your terminal:

```bash
# Predict with the default CatBoost models
lcqspr --input examples/data/sample.csv --output results/predictions.csv

# Predict with the bundled GNN model instead
lcqspr --input examples/data/sample.csv \
  --gnn-model models/GATv2Conv_phase_temp_cv0.pt \
  --output results/gnn_predictions.csv
```

You can also test a single molecule directly from the command line without creating a CSV file:

```bash
lcqspr --smiles "CCCCCCCC1C=CC(=CC=1)C2C=CC(=CC=2)C(=O)CCCCC" \
  --output results/single_prediction.csv
```

### Understanding the Output CSV Columns

The output CSV can be opened directly in Excel or pandas:

| Column pattern | Meaning |
|---|---|
| `SMILES` | Input molecular structure in SMILES format |
| `predicted_phase_sequence` | Reconstructed transition sequence in °C (e.g. `Cr 86.34 B 94.07 A 102.33 is`) |
| `pred_<Phase>type` | Binary decision (`1` = phase predicted to occur, `0` = absent; or class index for GNN) |
| `prob_<Phase>type` / `pos_conf_<Phase>type` | Predicted probability (`0.0`–`1.0`) that the molecule exhibits `<Phase>` |
| `pred_Melting`, `pred_Clearing` | Predicted melting and isotropic clearing temperatures (°C) |
| `pred_<Phase>m`, `pred_<Phase>p` | Predicted lower (`m`) and upper (`p`) transition temperatures (°C) for `<Phase>` |

---

## Notebook Guide

Every notebook in `notebooks/` is self-contained and executable from top to bottom. Before each code cell, a short **Caution** note highlights potential chemical or statistical pitfalls, and an **Extension** note explains how to adapt the step to your own dataset.

| Notebook | Audience & Purpose |
|---|---|
| [00_predict.ipynb](notebooks/00_predict.ipynb) | **Start here:** Side-by-side CatBoost and GNN inference from SMILES, molecular structure visualisation and CSV export |
| [01_prepare_data.ipynb](notebooks/01_prepare_data.ipynb) | Parse raw LiqCryst-style `Phases` strings, calculate Mordred descriptors, separate chiral (`*`) phases and apply quality filters |
| [02_train_catboost.ipynb](notebooks/02_train_catboost.ipynb) | 5-fold cross-validated CatBoost phase classification and multi-output temperature regression, feature importance and SHAP analysis |
| [03_train_lightgbm.ipynb](notebooks/03_train_lightgbm.ipynb) | Per-target LightGBM cross-validation, descriptor-family ablations (`Shape`, `Electric`, `Flexible`), learning curves and Optuna tuning |
| [04_train_gnn.ipynb](notebooks/04_train_gnn.ipynb) | Molecular graph construction, `GATv2Conv` / `NNConv` / `GPSConv` architectures, optional Laplacian positional encoding and joint training |
| [05_verify_dataset_filtering_and_paper_summary.ipynb](notebooks/05_verify_dataset_filtering_and_paper_summary.ipynb) | Step-by-step verification of dataset quality filtering (46,848 molecules), reproduction of `paper_summary.csv`, class-frequency vs F1 plot and matched `cv0` benchmark of the bundled CatBoost and GNN models |
| [06_evaluate_and_interpret.ipynb](notebooks/06_evaluate_and_interpret.ipynb) | Detailed classification reports, confusion matrices, hybrid phase/temperature model combination and Mann–Whitney U descriptor screening |

---

## Python Interface

### 1. Inference with bundled or custom models

```python
from lc_qspr import predict_smiles
from lc_qspr.gnn import predict_gnn_smiles

smiles_list = [
    "CCCCCCCCCC1=CC=C(C=C1)C2=CC=CC=C2",
]

# CatBoost prediction (uses bundled models by default)
cb_pred = predict_smiles(smiles_list)
print(cb_pred[["SMILES", "predicted_phase_sequence"]])

# GNN prediction (uses bundled GATv2Conv checkpoint by default)
gnn_pred = predict_gnn_smiles(smiles_list)
print(gnn_pred[["SMILES", "predicted_phase_sequence"]])
```

To use checkpoints from your own training run, pass their paths explicitly—no files in `models/` need to be overwritten:

```python
custom_cb = predict_smiles(
    smiles_list,
    phase_model="results/my_run/phase.cbm",
    temperature_model="results/my_run/temperature.cbm",
)

custom_gnn = predict_gnn_smiles(
    smiles_list,
    model_path="results/my_run/GNN_joint_cv0.pt",
)
```

The CLI accepts the same custom model options (`--phase-model`, `--temperature-model` and `--gnn-model`).

### 2. Preparing a dataset from SMILES and phase strings

```python
from lc_qspr.preprocessing import (
    add_mordred_descriptors,
    filter_training_table,
    parse_phase_table,
    read_table,
    separate_chiral_phases,
)

raw = read_table("examples/data/sample.csv")
parsed, report = parse_phase_table(raw)
described = add_mordred_descriptors(parsed)
separated, chiral_phases = separate_chiral_phases(described)
training_table = filter_training_table(
    separated,
    minimum_nitrogen_and_carbon=12,
)
```

### 3. Cross-validated model training

Training functions accept explicit feature lists, target names and fold indices rather than relying on global notebook state:

```python
from lc_qspr.catboost_training import CatBoostConfig, train_catboost_phases
from lc_qspr.evaluation import cross_validation_splits

splits = cross_validation_splits(
    training_table,
    phase_targets,
    mode="classification",
    n_splits=5,
)
result = train_catboost_phases(
    training_table,
    feature_names,
    phase_targets,
    splits,
    config=CatBoostConfig(iterations=5000),
)
```

Equivalent APIs are provided in `lc_qspr.lightgbm_training` and `lc_qspr.gnn` (supporting `classification`, `regression` and joint `both` modes).

### 4. Descriptor catalogue and physical families

`mordred_descriptors.csv` (grouping descriptors into `Shape`, `Electric` and `Flexible` physical families) and `varnames.csv` (the exact 1,613 descriptors expected by the bundled CatBoost models) are installed as package data:

```python
from lc_qspr import descriptor_catalogue, model_descriptor_names
from lc_qspr.descriptors import descriptor_families

catalogue = descriptor_catalogue()
families = descriptor_families()          # "All", "Shape", "Electric", "Flexible", etc.
descriptor_names = model_descriptor_names()
```

---

## Repository Structure

```text
LC_QSPR/
├── pyproject.toml              # Package metadata and optional dependency groups
├── src/lc_qspr/                # Reusable Python package
│   └── data/                   # Installed descriptor catalogue and variable lists
├── notebooks/                  # Step-by-step executable notebooks (00 to 06)
├── examples/data/              # Small repository-safe demonstration tables
├── models/                     # Distributed pre-trained CatBoost and GNN models
├── results/                    # Default destination for generated predictions and outputs
├── docs/                       # Migration map from the monolithic legacy notebook
├── tests/                      # Automated unit and notebook verification tests
└── obsolete/                   # Archived scripts and legacy files for provenance
```

---

## Citation

- **Yoshiaki Uchida, Shizuo Kaji and Naoto Nakano**, *Extracting Knowledge from Predictive Machine Learning Models*, **液晶 (Ekisho)**, Vol. 28, No. 4, pp. 234–241, 2024.
- **Yoshiaki Uchida, Shizuo Kaji, Naoto Nakano**, *Systematic Error Detection in the Database of Liquid Crystals (LiqCryst) Using Predictive Models*, **Soft Matter**, Issue 23, 2026. https://doi.org/10.1039/D6SM00257A
- **Yoshiaki Uchida, Shizuo Kaji and Naoto Nakano**, *Extracting the Physics of Liquid Crystalline Phase Transitions from a Predictive Machine-Learning Model: How Single-Molecule Structure Constrains Collective Phase Behavior*, preprint.

```bibtex
@article{uchida2026systematic,
  title   = {Systematic Error Detection in the Database of Liquid Crystals ({LiqCryst}) Using Predictive Models},
  author  = {Yoshiaki Uchida and Shizuo Kaji and Naoto Nakano},
  journal = {Soft Matter},
  number  = {23},
  year    = {2026},
  doi     = {10.1039/D6SM00257A},
  url     = {https://doi.org/10.1039/D6SM00257A},
}

@unpublished{uchida2026extracting,
  title  = {Extracting the Physics of Liquid Crystalline Phase Transitions from a Predictive Machine-Learning Model: How Single-Molecule Structure Constrains Collective Phase Behavior},
  author = {Yoshiaki Uchida and Shizuo Kaji and Naoto Nakano},
  note   = {Preprint},
}

@article{uchida_kaji_nakano_lcqspr,
  title   = {Extracting Knowledge from Predictive Machine Learning Models},
  author  = {Yoshiaki Uchida and Shizuo Kaji and Naoto Nakano},
  journal = {液晶},
  volume  = {28},
  number  = {4},
  pages   = {234--241},
  year    = {2024},
}
```
