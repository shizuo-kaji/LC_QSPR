# LC-QSPR

LC-QSPR predicts liquid-crystal phases and transition temperatures directly
from SMILES. It provides a workflow for data preparation, Mordred descriptors, CatBoost and LightGBM
training, graph neural networks, cross-validation, interpretation and inference.

The interface is designed for chemistry researchers: short notebooks explain
the scientific cautions before each computation, whilst reusable Python
functions keep long analyses consistent and testable.

## Installation

Python 3.10 or later is required. From the repository root, install the package
and the components needed for the work you intend to run:

```bash
# Prediction with the distributed CatBoost models
python -m pip install -e .

# Notebooks and conventional model training
python -m pip install -e ".[notebook,training]"

# Complete workflow, including PyTorch Geometric
python -m pip install -e ".[notebook,training,gnn]"
```

No GPU is required for inference. Full GNN training is considerably faster on a
compatible accelerator.

## Notebook guide


Loo at [00_predict.ipynb](notebooks/00_predict.ipynb) for a quick demonstration of the prediction workflow.

| Notebook | Purpose |
|---|---|
| `00_predict.ipynb` | Main: CatBoost and GNN inference from the same SMILES |
| `01_prepare_data.ipynb` | Parse phase strings, calculate descriptors, separate chiral phases and filter records |
| `02_train_catboost.ipynb` | Phase and temperature CV training, OOF results, importance and SHAP |
| `03_train_lightgbm.ipynb` | Per-target CV training, learning curves, importance and Optuna tuning |
| `04_train_gnn.ipynb` | Molecular graphs, GATv2Conv/NNConv, joint or single-task training and checkpoints |
| `06_evaluate_and_interpret.ipynb` | Metrics, hybrid result tables, paper results and Mann–Whitney screening |


## Quick prediction from CSV

Prepare a CSV containing a `SMILES` column, then run:

```bash
lcqspr --input examples/data/sample.csv --output results/predictions.csv
```

A SMILES string may also be supplied directly:

```bash
lcqspr --smiles "CCO" --output results/ethanol_prediction.csv
```

The output includes the readable `predicted_phase_sequence`, per-phase decisions
and probabilities, and all predicted lower and upper transition temperatures.

## Python interface

### CatBoost inference

```python
from lc_qspr import predict_smiles

prediction = predict_smiles([
    "CCCCCCCCCC1=CC=C(C=C1)C2=CC=CC=C2",
])
print(prediction[["SMILES", "predicted_phase_sequence"]])
```

### Selecting your own trained models

The distributed models remain the default. A CatBoost pair produced by your own
training run can be selected explicitly, without renaming or moving either file:

```python
prediction = predict_smiles(
    ["CCCCCCCCCC1=CC=C(C=C1)C2=CC=CC=C2"],
    phase_model="results/my_run/phase.cbm",
    temperature_model="results/my_run/temperature.cbm",
)
```

The command-line interface accepts the same paths:

```bash
lcqspr --input examples/data/sample.csv \
  --phase-model results/my_run/phase.cbm \
  --temperature-model results/my_run/temperature.cbm \
  --output results/my_run/predictions.csv
```

A user-trained GNN checkpoint is selected with one option:

```bash
lcqspr --input examples/data/sample.csv \
  --gnn-model results/my_run/GNN_joint_cv0.pt \
  --output results/my_run/gnn_predictions.csv
```

For Python-based GNN inference, pass the same checkpoint path to
`lc_qspr.gnn.predict_gnn_smiles`. Model paths may be relative to the current
working directory or represented by `pathlib.Path` objects.

### Data preparation

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
    minimum_nitrogen_and_carbon=0,
    suspicious_tokens=(),
)
```

### Model training

Training functions accept explicit feature names, target names and CV indices;
they do not rely on notebook globals.

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

Equivalent maintained APIs are provided in `lc_qspr.lightgbm_training` and
`lc_qspr.gnn`. The GNN training function supports `classification`, `regression`
and joint `both` modes.

## Package data

`mordred_descriptors.csv` and `varnames.csv` are installed as package data, so
they do not depend on the current working directory:

```python
from lc_qspr import descriptor_catalogue, model_descriptor_names

catalogue = descriptor_catalogue()
descriptor_names = model_descriptor_names()
```

## Repository structure

```text
LC_QSPR/
├── pyproject.toml              # Build metadata and dependency groups
├── src/lc_qspr/                # Reusable package
│   └── data/                   # Installed descriptor tables
├── notebooks/                  # Executable, task-focused workflows
├── examples/data/              # Small demonstration inputs
├── models/                     # Distributed CatBoost and GNN models
├── results/                    # New generated outputs
├── tests/                      # Unit and integration tests
└── obsolete/                   # Historical notebooks, code, data and results
```

## Citation

```bibtex
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
