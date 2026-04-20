# Quantitative Structure–Property Relationships for Liquid Crystals

Predict liquid-crystal **phase sequences** and **transition temperatures**
directly from SMILES using gradient-boosted trees and graph neural networks.

> Companion code for: **Machine Learning Analysis for the Contribution of
> Molecular Fine Structures to the Symmetry of Liquid Crystals** —
> Yoshiaki Uchida, Shizuo Kaji, Naoto Nakano.

---

## What is included

| Path                                 | Description                                                      |
| ------------------------------------ | ---------------------------------------------------------------- |
| `LC_QSPR.ipynb`                      | Main Jupyter notebook (inference, training, dataset preparation) |
| `sample.csv` / `sample.parquet`      | Tiny example dataset                                             |
| `sample_desc.csv`                    | Pre-computed Mordred descriptors for `sample.csv`                |
| `mordred_descriptors.csv`            | Descriptor catalogue (Shape / Electric / Flexible families)      |
| `varnames.csv`                       | Descriptor names used by the bundled CatBoost models             |
| `CatBoost_phase.cbm`                 | Pre-trained CatBoost phase classifier                            |
| `CatBoost_temperature_7000epochs.cbm`| Pre-trained CatBoost transition-temperature regressor            |
| `GATv2Conv_phase_temp_cv0.pt`        | Pre-trained GATv2Conv GNN (joint phase + temperature), portable `state_dict` format |
| `LiqCryst52.parquet`                 | Full descriptor database used in the paper (large)               |

---

## Installation

Python ≥ 3.10 is recommended. Create an isolated environment and install
the dependencies — a GPU is optional (the notebook falls back to CPU).

```bash
# 1. Create an environment (conda or venv — either is fine)
conda create -n lcqspr python=3.11 -y
conda activate lcqspr

# 2. Core scientific stack
pip install numpy pandas matplotlib seaborn tqdm pyarrow joblib scipy scikit-learn ipywidgets

# 3. Cheminformatics & descriptors
pip install rdkit mordred iterative-stratification

# 4. Gradient boosting
pip install catboost lightgbm shap

# 5. (Optional) hyper-parameter tuning + statistical plots
pip install optuna statannotations

# 6. PyTorch + PyTorch Geometric
#    Pick the wheel index that matches your CUDA version — see
#    https://pytorch.org and https://pytorch-geometric.readthedocs.io
pip install torch torcheval torch_geometric
```

> If you prefer **conda** for CUDA-enabled PyTorch / PyG:
>
> ```bash
> conda install -c pytorch -c nvidia pytorch torchvision torchaudio torcheval pytorch-cuda=12.1
> conda install -c pyg pyg
> ```

---

## Quick start (inference)

If you only want to predict phases for a few molecules, you do not need to
train anything — the bundled checkpoints are enough.

1. Launch the notebook:

   ```bash
   jupyter lab LC_QSPR.ipynb
   ```

2. Run **Section 2 (Imports)**, **Section 3 (Global settings)** and
   **Section 4 (GNN building blocks)**.

3. Jump to **Section 5 — Inference with pre-trained models**, edit the
   `SMILES` list, and run the CatBoost and/or GNN prediction cells.

The output is a `DataFrame` with a reconstructed LiqCryst-style
`pred_Phases` column plus per-target probabilities and temperatures.

---

## Training your own models

1. **Prepare a dataset.** See **Section 7** of the notebook.

   The input is a CSV with at least the columns:

   - `ID` — integer identifier
   - `SMILES` — SMILES string
   - `Phases` — LiqCryst-style phase sequence, e.g.
     `Cr 76.8 A 112.5 N* 119.8 is`

   A small `sample.csv` is bundled for quick testing. Section 7 parses
   the `Phases` column into per-phase type / T⁻ / T⁺ columns and then
   computes ~1,800 Mordred descriptors in parallel, writing the result
   to a Parquet file.

2. **Train.** Point `descriptor_fn` in **Section 6.1** at the parquet
   file you just created, then run:

   - **Section 6.2** to train CatBoost (phase classifier + temperature
     regressor),
   - **Section 6.3** to train LightGBM,
   - **Section 6.4** to train a GNN jointly for phase and temperature.

   Each section performs 5-fold cross-validation and writes
   out-of-fold predictions, feature importances and summary metrics
   into `outdir` (default `result_test/`).

---

## Project structure

```
LC_QSPR/
├── LC_QSPR.ipynb             # main notebook
├── README.md
├── mordred_descriptors.csv   # descriptor catalogue
├── varnames.csv              # descriptors used by the CatBoost models
├── sample.csv                # tiny example dataset
├── sample_desc.csv           # with pre-computed descriptors
├── CatBoost_*.cbm            # pre-trained gradient-boosted models
├── GATv2Conv_*.pth           # pre-trained GNN
├── data/                     # raw LiqCryst-derived tables
├── images/                   # figures generated for the paper
├── reference/                # reference outputs
└── results/                  # training logs / model outputs
```

---

## Citation

If you use this code please cite:

```bibtex
@article{uchida_kaji_nakano_lcqspr,
  title  = {Machine Learning Analysis for the Contribution of Molecular
            Fine Structures to the Symmetry of Liquid Crystals},
  author = {Uchida, Yoshiaki and Kaji, Shizuo and Nakano, Naoto},
}
```

## License

See the repository for license terms.
