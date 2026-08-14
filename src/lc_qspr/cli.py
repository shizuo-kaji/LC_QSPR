"""Command-line interface intended for small, routine predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .prediction import predict_smiles


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lcqspr",
        description="Predict liquid-crystal phases and temperatures from SMILES.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smiles", action="append", help="SMILES string; repeat for multiple molecules")
    source.add_argument("--input", type=Path, help="CSV file containing a SMILES column")
    parser.add_argument("--output", type=Path, default=Path("results/predictions.csv"))
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing CatBoost models with the distributed filenames.",
    )
    parser.add_argument(
        "--phase-model",
        type=Path,
        help="User-trained CatBoost phase-model file; use with --temperature-model.",
    )
    parser.add_argument(
        "--temperature-model",
        type=Path,
        help="User-trained CatBoost temperature-model file; use with --phase-model.",
    )
    parser.add_argument(
        "--gnn-model",
        type=Path,
        help="GNN checkpoint file; selects GNN prediction instead of CatBoost.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for CatBoost phase decisions.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.input:
        table = pd.read_csv(args.input)
        if "SMILES" not in table:
            raise SystemExit("Input CSV must contain a column named 'SMILES'.")
        smiles = table["SMILES"].astype(str).tolist()
    else:
        smiles = args.smiles

    if args.gnn_model is not None:
        if args.phase_model is not None or args.temperature_model is not None:
            raise SystemExit(
                "--gnn-model cannot be combined with --phase-model or "
                "--temperature-model."
            )
        try:
            from .gnn import predict_gnn_smiles
        except ModuleNotFoundError as error:
            raise SystemExit(
                "GNN prediction requires the optional GNN dependencies. "
                "Install the package with the 'gnn' extra."
            ) from error
        prediction = predict_gnn_smiles(smiles, args.gnn_model)
    else:
        if (args.phase_model is None) != (args.temperature_model is None):
            raise SystemExit(
                "--phase-model and --temperature-model must be supplied together."
            )
        prediction = predict_smiles(
            smiles,
            args.model_dir,
            phase_model=args.phase_model,
            temperature_model=args.temperature_model,
            threshold=args.threshold,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(args.output, index=True)
    print(prediction[["SMILES", "predicted_phase_sequence"]].to_string(index=False))
    print(f"\nSaved {len(prediction)} prediction(s) to {args.output}")


if __name__ == "__main__":
    main()
