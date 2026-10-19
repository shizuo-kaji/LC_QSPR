"""PyTorch Geometric models, molecular graphs, training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys
import time
import types
from typing import Any, Sequence
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch import nn
import torch.nn.functional as functional
import torch_geometric.nn
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    NNConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from .constants import CHIRAL_PHASE_TYPES, PERMITTED_ATOMS, TEMPERATURE_TARGETS
from .phases import format_phase_sequence


def one_hot_encoding(value: Any, permitted: Sequence[Any]) -> list[int]:
    """Encode unknown values as the final member of ``permitted``."""
    if value not in permitted:
        value = permitted[-1]
    return [int(value == member) for member in permitted]


#: Alternative spellings accepted by the attribute exclusion lists.
_EXCLUSION_ALIASES = {
    "n_heavy_neighbours": ("n_heavy_neighbors",),
    "bond_is_conjugated": ("bond_is_conj",),
}


def _is_excluded(name: str, exclude: Sequence[str]) -> bool:
    """True when ``name`` or one of its accepted aliases is excluded."""
    if name in exclude:
        return True
    return any(alias in exclude for alias in _EXCLUSION_ALIASES.get(name, ()))


@dataclass(frozen=True)
class FeatureBlock:
    """One contiguous, individually ablatable block of a feature vector.

    ``kind`` records how a block should be filled when an ablation keeps the
    slot but removes its information: ``one-hot`` blocks take a random unit
    vector, ``boolean`` blocks a fair coin per column, and ``scalar`` blocks a
    standard normal draw. This keeps a randomised slot on the same support as
    the attribute it replaces.
    """

    name: str
    width: int
    kind: str


#: Node feature blocks in the exact order emitted by :func:`atom_features`.
#: ``atom_identity`` is controlled by ``atom_encoding`` rather than by name.
NODE_FEATURE_BLOCKS: tuple[FeatureBlock, ...] = (
    FeatureBlock("atom_identity", len(PERMITTED_ATOMS), "one-hot"),
    FeatureBlock("n_heavy_neighbours", 6, "one-hot"),
    FeatureBlock("n_valence", 8, "one-hot"),
    FeatureBlock("formal_charge", 8, "one-hot"),
    FeatureBlock("hybridisation_type", 7, "one-hot"),
    FeatureBlock("is_in_a_ring", 1, "boolean"),
    FeatureBlock("is_aromatic", 1, "boolean"),
    FeatureBlock("atomic_mass_scaled", 1, "scalar"),
    FeatureBlock("vdw_radius_scaled", 1, "scalar"),
    FeatureBlock("covalent_radius_scaled", 1, "scalar"),
    FeatureBlock("chirality_type", 4, "one-hot"),
    FeatureBlock("n_hydrogens", 6, "one-hot"),
)

#: Edge feature blocks in the exact order emitted by :func:`bond_features`.
EDGE_FEATURE_BLOCKS: tuple[FeatureBlock, ...] = (
    FeatureBlock("bond_type", 4, "one-hot"),
    FeatureBlock("bond_is_conjugated", 1, "boolean"),
    FeatureBlock("bond_is_in_ring", 1, "boolean"),
    FeatureBlock("stereochemistry", 4, "one-hot"),
)

FULL_NODE_FEATURE_WIDTH = sum(block.width for block in NODE_FEATURE_BLOCKS)
FULL_EDGE_FEATURE_WIDTH = sum(block.width for block in EDGE_FEATURE_BLOCKS)


def ablated_feature_spans(
    blocks: Sequence[FeatureBlock],
    exclude: Sequence[str] = (),
    *,
    atom_encoding: str = "one-hot",
) -> tuple[tuple[int, int, str], ...]:
    """Return ``(start, stop, kind)`` for every ablated block of ``blocks``.

    The offsets index the full-width feature vector produced with
    ``placeholder_excluded=True``, so they identify exactly the columns that
    carry a placeholder rather than a measured attribute.
    """
    spans: list[tuple[int, int, str]] = []
    offset = 0
    for block in blocks:
        if block.name == "atom_identity":
            excluded = atom_encoding != "one-hot"
        else:
            excluded = _is_excluded(block.name, exclude)
        if excluded:
            spans.append((offset, offset + block.width, block.kind))
        offset += block.width
    return tuple(spans)


def randomise_ablated_slots(
    values: torch.Tensor,
    spans: tuple[tuple[int, int, str], ...],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Refill ablated feature columns with information-free noise, in place.

    Each ablated column is redrawn independently for every node or edge, on the
    support of the attribute it replaces: a random unit vector for a one-hot
    block, a fair coin for a boolean flag and a standard normal for a scaled
    scalar. Redrawing on every call means the noise cannot encode the molecule,
    so the ablated attribute contributes no information; what it does contribute
    is a distinct random value per node, which breaks the permutation symmetry
    that an all-constant input would otherwise impose.
    """
    if not spans or values.numel() == 0:
        return values
    rows = values.shape[0]
    for start, stop, kind in spans:
        width = stop - start
        if kind == "one-hot":
            choice = torch.randint(
                width, (rows,), device=values.device, generator=generator
            )
            values[:, start:stop] = functional.one_hot(choice, width).to(values.dtype)
        elif kind == "boolean":
            values[:, start:stop] = torch.randint(
                2, (rows, width), device=values.device, generator=generator
            ).to(values.dtype)
        else:
            values[:, start:stop] = torch.randn(
                (rows, width),
                device=values.device,
                dtype=values.dtype,
                generator=generator,
            )
    return values


def apply_slot_noise(
    batch,
    node_spans: tuple[tuple[int, int, str], ...],
    edge_spans: tuple[tuple[int, int, str], ...],
    generator: torch.Generator | None = None,
):
    """Refill the ablated node and edge columns of one batch."""
    randomise_ablated_slots(batch.x, node_spans, generator)
    randomise_ablated_slots(batch.edge_attr, edge_spans, generator)
    return batch

def _block_appender(
    features: list[float], exclude: Sequence[str], placeholder_excluded: bool
):
    """Return a helper that appends, zero-fills or omits one feature block.

    With ``placeholder_excluded`` an excluded block keeps its slots and receives
    zeros, so the feature vector always has its full registered width and the
    ablated columns can be located by :func:`ablated_feature_spans`. Without it
    the block is omitted and the vector shortens, which is the historical
    behaviour.
    """

    def append(name: str, width: int, values) -> None:
        if not _is_excluded(name, exclude):
            block = values()
            if len(block) != width:
                raise ValueError(
                    f"Feature block {name} produced {len(block)} values, "
                    f"but the registry declares {width}."
                )
            features.extend(block)
        elif placeholder_excluded:
            features.extend([0.0] * width)

    return append


def atom_features(
    atom,
    *,
    atom_encoding: str = "one-hot",
    exclude: Sequence[str] = ("n_heavy_neighbours",),
    placeholder_excluded: bool = False,
) -> tuple[np.ndarray, int | None]:
    """Return continuous/categorical atom features and an optional atom ID."""
    if atom_encoding not in {"one-hot", "integer", "none"}:
        raise ValueError("atom_encoding must be 'one-hot', 'integer' or 'none'.")
    atom_lookup = {symbol: index for index, symbol in enumerate(PERMITTED_ATOMS)}
    symbol = atom.GetSymbol() if atom.GetSymbol() in atom_lookup else "Unknown"
    features: list[float] = []
    atom_id = None
    if atom_encoding == "one-hot":
        features.extend(one_hot_encoding(symbol, PERMITTED_ATOMS))
    elif atom_encoding == "integer":
        atom_id = atom_lookup[symbol]
    elif placeholder_excluded:
        features.extend([0.0] * len(PERMITTED_ATOMS))

    append = _block_appender(features, exclude, placeholder_excluded)
    append(
        "n_heavy_neighbours",
        6,
        lambda: one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, "MoreThanFour"]),
    )
    append(
        "n_valence",
        8,
        lambda: one_hot_encoding(
            atom.GetTotalValence(), [0, 1, 2, 3, 4, 5, 6, "MoreThanSix"]
        ),
    )
    append(
        "formal_charge",
        8,
        lambda: one_hot_encoding(
            atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3, "Extreme"]
        ),
    )
    append(
        "hybridisation_type",
        7,
        lambda: one_hot_encoding(
            str(atom.GetHybridization()),
            ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"],
        ),
    )
    append("is_in_a_ring", 1, lambda: [int(atom.IsInRing())])
    append("is_aromatic", 1, lambda: [int(atom.GetIsAromatic())])
    append(
        "atomic_mass_scaled", 1, lambda: [float((atom.GetMass() - 10.812) / 116.092)]
    )
    append(
        "vdw_radius_scaled",
        1,
        lambda: [
            float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)
        ],
    )
    append(
        "covalent_radius_scaled",
        1,
        lambda: [
            float(
                (Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)
                / 0.76
            )
        ],
    )
    append(
        "chirality_type",
        4,
        lambda: one_hot_encoding(
            str(atom.GetChiralTag()),
            [
                "CHI_UNSPECIFIED",
                "CHI_TETRAHEDRAL_CW",
                "CHI_TETRAHEDRAL_CCW",
                "CHI_OTHER",
            ],
        ),
    )
    append(
        "n_hydrogens",
        6,
        lambda: one_hot_encoding(
            atom.GetTotalNumHs(), [0, 1, 2, 3, 4, "MoreThanFour"]
        ),
    )
    if not features:
        features = [0.0]
    return np.asarray(features, dtype=np.float32), atom_id


def bond_features(
    bond, *, exclude: Sequence[str] = (), placeholder_excluded: bool = False
) -> np.ndarray:
    """Return edge features for one RDKit bond."""
    features: list[float] = []
    append = _block_appender(features, exclude, placeholder_excluded)
    append(
        "bond_type",
        4,
        lambda: one_hot_encoding(
            bond.GetBondType(),
            [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC,
            ],
        ),
    )
    append("bond_is_conjugated", 1, lambda: [int(bond.GetIsConjugated())])
    append("bond_is_in_ring", 1, lambda: [int(bond.IsInRing())])
    append(
        "stereochemistry",
        4,
        lambda: one_hot_encoding(
            str(bond.GetStereo()),
            ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"],
        ),
    )
    return np.asarray(features if features else [0.0], dtype=np.float32)


def _laplacian_positional_encoding(
    molecule: Chem.Mol, dimensions: int
) -> np.ndarray:
    """Return deterministic symmetric-normalised Laplacian eigenvectors.

    The Laplacian is constructed from the simple undirected molecular graph:
    every bonded atom pair has adjacency weight one, irrespective of bond
    order, aromaticity or other bond attributes.
    """
    if dimensions < 0:
        raise ValueError("laplacian_pe_dimension must be non-negative.")
    num_atoms = molecule.GetNumAtoms()
    if dimensions == 0:
        return np.empty((num_atoms, 0), dtype=np.float32)
    adjacency = np.zeros((num_atoms, num_atoms), dtype=np.float64)
    for bond in molecule.GetBonds():
        source = bond.GetBeginAtomIdx()
        destination = bond.GetEndAtomIdx()
        adjacency[source, destination] = 1.0
        adjacency[destination, source] = 1.0
    degree = adjacency.sum(axis=1)
    inverse_sqrt_degree = np.zeros_like(degree)
    connected = degree > 0
    inverse_sqrt_degree[connected] = degree[connected] ** -0.5
    laplacian = -(
        inverse_sqrt_degree[:, None]
        * adjacency
        * inverse_sqrt_degree[None, :]
    )
    laplacian[np.diag_indices(num_atoms)] = connected.astype(float)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)]
    positional_encoding = eigenvectors[:, 1 : dimensions + 1]
    if positional_encoding.size:
        pivots = np.abs(positional_encoding).argmax(axis=0)
        signs = np.sign(
            positional_encoding[
                pivots, np.arange(positional_encoding.shape[1])
            ]
        )
        signs[signs == 0] = 1
        positional_encoding *= signs
    if positional_encoding.shape[1] < dimensions:
        positional_encoding = np.pad(
            positional_encoding,
            ((0, 0), (0, dimensions - positional_encoding.shape[1])),
        )
    return positional_encoding.astype(np.float32, copy=False)


def smiles_to_graph(
    smiles: str,
    *,
    atom_encoding: str = "one-hot",
    exclude: Sequence[str] = (),
    placeholder_excluded: bool = False,
    explicit_hydrogens: bool = False,
    laplacian_pe_dimension: int = 0,
    y_float: Sequence[float] | None = None,
    y_long: Sequence[int] | None = None,
) -> Data:
    """Convert one SMILES string into a bidirectional PyG graph."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    if explicit_hydrogens:
        molecule = Chem.AddHs(molecule)
    laplacian_pe = _laplacian_positional_encoding(
        molecule, laplacian_pe_dimension
    )
    node_features = []
    atom_ids = []
    for atom in molecule.GetAtoms():
        features, atom_id = atom_features(
            atom,
            atom_encoding=atom_encoding,
            exclude=exclude,
            placeholder_excluded=placeholder_excluded,
        )
        node_features.append(features)
        if atom_id is not None:
            atom_ids.append(atom_id)
    rows, columns = np.nonzero(Chem.GetAdjacencyMatrix(molecule))
    edge_features = [
        bond_features(
            molecule.GetBondBetweenAtoms(int(source), int(destination)),
            exclude=exclude,
            placeholder_excluded=placeholder_excluded,
        )
        for source, destination in zip(rows, columns)
    ]
    data = Data(
        x=torch.tensor(np.asarray(node_features), dtype=torch.float32),
        edge_index=torch.tensor(np.vstack([rows, columns]), dtype=torch.long),
        edge_attr=torch.tensor(np.asarray(edge_features), dtype=torch.float32),
        y=torch.tensor(
            [list(y_float) if y_float is not None else []], dtype=torch.float32
        ),
        z=torch.tensor([list(y_long) if y_long is not None else []], dtype=torch.long),
    )
    if atom_ids:
        data.node_id = torch.tensor(atom_ids, dtype=torch.long)
    if laplacian_pe_dimension:
        data.laplacian_pe = torch.tensor(laplacian_pe, dtype=torch.float32)
    return data


class LiquidCrystalGraphDataset(InMemoryDataset):
    """In-memory molecular graphs with optional phase and temperature targets."""

    def __init__(
        self,
        smiles: Sequence[str] | pd.Series,
        *,
        classification_targets: pd.DataFrame | None = None,
        regression_targets: pd.DataFrame | None = None,
        atom_encoding: str = "one-hot",
        exclude: Sequence[str] = (),
        placeholder_excluded: bool = False,
        explicit_hydrogens: bool = False,
        laplacian_pe_dimension: int = 0,
        transform=None,
    ):
        super().__init__(root=None, transform=transform)
        smiles = list(smiles)
        self.smiles = smiles
        self.laplacian_pe_dimension = laplacian_pe_dimension
        row_count = len(smiles)
        if classification_targets is None:
            classification_targets = pd.DataFrame(index=range(row_count))
        else:
            classification_targets = classification_targets.reset_index(drop=True)
        if regression_targets is None:
            regression_targets = pd.DataFrame(index=range(row_count))
        else:
            regression_targets = regression_targets.reset_index(drop=True)
        if len(classification_targets) != row_count or len(regression_targets) != row_count:
            raise ValueError("Target tables must contain one row per SMILES string.")

        self.classification_names = list(classification_targets.columns)
        self.regression_names = list(regression_targets.columns)
        self.class_counts = [
            max(int(classification_targets[column].max()) + 1, 2)
            for column in self.classification_names
        ]
        self.output_channel_masks = []
        offset = 0
        total_classes = sum(self.class_counts)
        for class_count in self.class_counts:
            mask = [False] * total_classes
            mask[offset : offset + class_count] = [True] * class_count
            self.output_channel_masks.append(mask)
            offset += class_count
        self.output_channels = []
        if self.classification_names:
            self.output_channels.append(total_classes)
        if self.regression_names:
            self.output_channels.append(len(self.regression_names))

        graphs = []
        for index, value in enumerate(smiles):
            graph = smiles_to_graph(
                value,
                atom_encoding=atom_encoding,
                exclude=exclude,
                placeholder_excluded=placeholder_excluded,
                explicit_hydrogens=explicit_hydrogens,
                laplacian_pe_dimension=laplacian_pe_dimension,
                y_float=regression_targets.iloc[index].to_numpy(dtype=float),
                y_long=classification_targets.iloc[index].to_numpy(dtype=int),
            )
            graphs.append(graph)
        if not graphs:
            raise ValueError("At least one SMILES string is required.")
        self.data, self.slices = self.collate(graphs)
        self.dim_node_attr = graphs[0].x.shape[1]
        self.dim_edge_attr = graphs[0].edge_attr.shape[1]


class GNN(nn.Module):
    """Multi-head, edge-aware molecular graph neural network.

    ``pooling="mean-max"`` and ``residual=False`` preserve the historical
    architecture used by the distributed checkpoint.  Size-aware and sum
    pooling retain graph-size information, whilst residual connections can
    stabilise deep message-passing stacks.  ``conv_type="GPSConv"`` combines
    a local edge-aware GINE layer with global multi-head attention.  A positive
    ``laplacian_pe_dimension`` appends precomputed simple-graph Laplacian
    eigenvectors to the node inputs.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        num_layers: int = 5,
        out_channels: Sequence[int] = (1, 1),
        dim_node_attr: int = -1,
        dim_edge_attr: int = -1,
        dropout: float = 0.0,
        id_emb_dim: int = 0,
        num_heads: int = 4,
        conv_type: str = "GATv2Conv",
        pooling: str = "mean-max",
        residual: bool = False,
        laplacian_pe_dimension: int = 0,
    ):
        super().__init__()
        if pooling not in {"mean-max", "mean-max-size", "mean-max-sum"}:
            raise ValueError(
                "pooling must be 'mean-max', 'mean-max-size' or 'mean-max-sum'."
            )
        if conv_type == "GPSConv" and (
            num_heads < 1 or hidden_channels % num_heads != 0
        ):
            raise ValueError(
                "GPSConv requires a positive number of heads that divides "
                "hidden_channels."
            )
        if laplacian_pe_dimension < 0:
            raise ValueError("laplacian_pe_dimension must be non-negative.")
        self.dropout = dropout
        self.conv_type = conv_type
        self.pooling = pooling
        self.residual = residual
        self.hidden_channels = hidden_channels
        self.dim_node_attr = dim_node_attr
        self.dim_edge_attr = dim_edge_attr
        self.num_heads = num_heads
        self.laplacian_pe_dimension = laplacian_pe_dimension
        effective_node_channels = dim_node_attr + laplacian_pe_dimension
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        self.node_projection = (
            nn.Linear(effective_node_channels, hidden_channels)
            if conv_type == "GPSConv"
            else None
        )
        for layer in range(num_layers):
            input_channels = (
                hidden_channels
                if conv_type == "GPSConv"
                else effective_node_channels if layer == 0 else hidden_channels
            )
            if conv_type == "GPSConv":
                update_network = nn.Sequential(
                    nn.Linear(hidden_channels, 2 * hidden_channels),
                    nn.GELU(),
                    nn.Linear(2 * hidden_channels, hidden_channels),
                )
                local_convolution = torch_geometric.nn.GINEConv(
                    update_network,
                    edge_dim=dim_edge_attr,
                    train_eps=True,
                )
                convolution = torch_geometric.nn.GPSConv(
                    hidden_channels,
                    local_convolution,
                    heads=num_heads,
                    dropout=dropout,
                    act="gelu",
                    norm="layer_norm",
                )
            elif conv_type == "NNConv":
                edge_network = nn.Sequential(
                    nn.Linear(dim_edge_attr, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels * input_channels),
                )
                convolution = NNConv(
                    input_channels, hidden_channels, edge_network, aggr="mean"
                )
            else:
                convolution_class = getattr(torch_geometric.nn, conv_type)
                convolution = convolution_class(
                    input_channels,
                    hidden_channels // num_heads,
                    edge_dim=dim_edge_attr,
                    heads=num_heads,
                )
            self.convs.append(convolution)
            self.bns.append(
                nn.Identity()
                if conv_type == "GPSConv"
                else nn.BatchNorm1d(hidden_channels)
            )
            if residual:
                self.residual_projections.append(
                    nn.Identity()
                    if input_channels == hidden_channels
                    else nn.Linear(input_channels, hidden_channels, bias=False)
                )
        self.emb = (
            nn.Embedding(len(PERMITTED_ATOMS), id_emb_dim) if id_emb_dim > 0 else None
        )
        pooled_channels = (
            3 * hidden_channels
            if pooling == "mean-max-sum"
            else 2 * hidden_channels + (1 if pooling == "mean-max-size" else 0)
        )
        self.fc1 = nn.ModuleList(
            [nn.Linear(pooled_channels, 2 * hidden_channels) for _ in out_channels]
        )
        self.fc2 = nn.ModuleList(
            [nn.Linear(2 * hidden_channels, output) for output in out_channels]
        )

    def forward(self, data) -> list[torch.Tensor]:
        node_values = data.x
        if self.emb is not None:
            if not hasattr(data, "node_id"):
                raise ValueError("Integer atom encoding is required by this model.")
            node_values = torch.cat([self.emb(data.node_id), node_values], dim=1)
        laplacian_pe_dimension = getattr(self, "laplacian_pe_dimension", 0)
        if laplacian_pe_dimension:
            if not hasattr(data, "laplacian_pe"):
                raise ValueError(
                    "This model requires Laplacian positional encodings."
                )
            if data.laplacian_pe.shape[1] != laplacian_pe_dimension:
                raise ValueError(
                    "Laplacian positional encoding width does not match the model."
                )
            node_values = torch.cat([node_values, data.laplacian_pe], dim=1)
        node_projection = getattr(self, "node_projection", None)
        if node_projection is not None:
            node_values = functional.gelu(node_projection(node_values))
        if self.conv_type == "GPSConv":
            for convolution, residual_projection in zip(
                self.convs,
                (
                    self.residual_projections
                    if getattr(self, "residual", False)
                    else [None] * len(self.convs)
                ),
                strict=True,
            ):
                updated = convolution(
                    node_values,
                    data.edge_index,
                    batch=data.batch,
                    edge_attr=data.edge_attr,
                )
                if residual_projection is not None:
                    updated = (
                        updated + residual_projection(node_values)
                    ) / np.sqrt(2.0)
                node_values = updated
        elif getattr(self, "residual", False):
            layer_iterator = zip(
                self.convs, self.bns, self.residual_projections, strict=True
            )
            for convolution, batch_normalisation, residual_projection in layer_iterator:
                updated = batch_normalisation(
                    convolution(node_values, data.edge_index, data.edge_attr)
                )
                node_values = functional.elu(
                    (updated + residual_projection(node_values)) / np.sqrt(2.0)
                )
        else:
            for convolution, batch_normalisation in zip(
                self.convs, self.bns, strict=True
            ):
                node_values = functional.elu(
                    batch_normalisation(
                        convolution(node_values, data.edge_index, data.edge_attr)
                    )
                )
        pooled_values = [
            global_mean_pool(node_values, data.batch),
            global_max_pool(node_values, data.batch),
        ]
        pooling = getattr(self, "pooling", "mean-max")
        if pooling == "mean-max-sum":
            pooled_values.append(global_add_pool(node_values, data.batch))
        elif pooling == "mean-max-size":
            graph_size = torch.bincount(data.batch).to(node_values.dtype)
            pooled_values.append(torch.log1p(graph_size).unsqueeze(1))
        pooled = torch.cat(pooled_values, dim=1)
        outputs = []
        for first_layer, second_layer in zip(self.fc1, self.fc2):
            values = functional.elu(first_layer(pooled))
            if self.dropout > 0:
                values = functional.dropout(
                    values, p=self.dropout, training=self.training
                )
            outputs.append(second_layer(values))
        return outputs


def save_gnn(model: GNN, path: str | Path, metadata: dict | None = None) -> None:
    """Save a portable state dictionary together with reconstruction settings."""
    first_convolution = model.convs[0]
    if model.conv_type == "GPSConv":
        hidden_channels = model.hidden_channels
        dim_node_attr = model.dim_node_attr
        dim_edge_attr = model.dim_edge_attr
        num_heads = model.num_heads
    elif model.conv_type == "NNConv":
        hidden_channels = first_convolution.out_channels
        dim_node_attr = first_convolution.in_channels
        dim_edge_attr = first_convolution.nn[0].in_features
        num_heads = 1
    else:
        num_heads = getattr(first_convolution, "heads", 1)
        hidden_channels = num_heads * first_convolution.out_channels
        dim_node_attr = first_convolution.in_channels
        dim_edge_attr = getattr(first_convolution, "edge_dim", -1) or -1
    dim_node_attr = getattr(model, "dim_node_attr", dim_node_attr)
    configuration = {
        "hidden_channels": int(hidden_channels),
        "num_layers": len(model.convs),
        "out_channels": [layer.out_features for layer in model.fc2],
        "dim_node_attr": int(dim_node_attr),
        "dim_edge_attr": int(dim_edge_attr),
        "dropout": float(model.dropout),
        "id_emb_dim": model.emb.embedding_dim if model.emb is not None else 0,
        "num_heads": int(num_heads),
        "conv_type": model.conv_type,
        "pooling": getattr(model, "pooling", "mean-max"),
        "residual": getattr(model, "residual", False),
        "laplacian_pe_dimension": getattr(
            model, "laplacian_pe_dimension", 0
        ),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": configuration, "state_dict": model.state_dict(), "meta": metadata or {}},
        path,
    )


def load_gnn(path: str | Path, map_location=None) -> GNN:
    """Load the portable format, with a fallback for historical full pickles."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"GNN checkpoint was not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        if isinstance(checkpoint, dict) and {"config", "state_dict"} <= checkpoint.keys():
            model = GNN(**checkpoint["config"])
            model.load_state_dict(checkpoint["state_dict"])
            return model
    except Exception:
        pass

    try:
        import torch_geometric.nn.conv.utils.inspector  # noqa: F401
    except ModuleNotFoundError:
        from torch_geometric import inspector as pyg_inspector

        for module_name in (
            "torch_geometric.nn.conv.utils",
            "torch_geometric.nn.conv.utils.inspector",
        ):
            sys.modules.setdefault(module_name, types.ModuleType(module_name))
        sys.modules["torch_geometric.nn.conv.utils.inspector"] = pyg_inspector
    import torch_geometric.nn.conv.message_passing as message_passing

    original_setstate = message_passing.MessagePassing.__setstate__
    message_passing.MessagePassing.__setstate__ = (
        lambda self, state: self.__dict__.update(state)
    )
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    finally:
        message_passing.MessagePassing.__setstate__ = original_setstate


class FocalLoss(nn.Module):
    """Focal loss for imbalanced multi-class targets."""

    def __init__(self, gamma: float = 1.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        cross_entropy = functional.cross_entropy(
            inputs, targets.reshape(-1), reduction="none"
        )
        probability = torch.exp(-cross_entropy)
        return (((1 - probability) ** self.gamma) * cross_entropy).mean()


@dataclass
class EpochMetrics:
    loss: float
    accuracy: list[float] = field(default_factory=list)
    f1: list[float] = field(default_factory=list)
    mae: list[float] = field(default_factory=list)


def _classification_metrics(
    logits: torch.Tensor, truth: torch.Tensor
) -> tuple[float, float]:
    prediction = logits.argmax(dim=1)
    truth = truth.reshape(-1)
    accuracy = (prediction == truth).float().mean().item()
    classes = range(logits.shape[1])
    class_f1 = []
    for class_index in classes:
        predicted_positive = prediction == class_index
        actual_positive = truth == class_index
        true_positive = (predicted_positive & actual_positive).sum().float()
        denominator = predicted_positive.sum() + actual_positive.sum()
        if denominator > 0:
            class_f1.append((2 * true_positive / denominator).item())
    return accuracy, float(np.mean(class_f1)) if class_f1 else np.nan


def _batch_loss(
    outputs: Sequence[torch.Tensor],
    batch,
    mode: str,
    class_masks: Sequence[Sequence[bool]],
    classification_loss,
    regression_loss,
) -> torch.Tensor:
    loss = outputs[0].sum() * 0
    if mode in {"classification", "both"}:
        for target_index, mask in enumerate(class_masks):
            loss = loss + classification_loss(
                outputs[0][:, mask], batch.z[:, target_index]
            )
    if mode in {"regression", "both"}:
        output_index = 0 if mode == "regression" else 1
        observed = ~torch.isnan(batch.y)
        if observed.any():
            loss = loss + regression_loss(outputs[output_index][observed], batch.y[observed])
    return loss


def run_gnn_epoch(
    model: GNN,
    loader: DataLoader,
    *,
    mode: str,
    class_masks: Sequence[Sequence[bool]] = (),
    optimiser=None,
    scheduler=None,
    device: str | torch.device = "cpu",
) -> EpochMetrics:
    """Run one training or evaluation epoch without relying on global variables."""
    training = optimiser is not None
    model.train(training)
    total_loss = 0.0
    total_rows = 0
    accuracy_values = [[] for _ in class_masks]
    f1_values = [[] for _ in class_masks]
    mae_values: list[list[float]] | None = None
    classification_loss = FocalLoss()
    regression_loss = nn.L1Loss()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            if training:
                optimiser.zero_grad()
            outputs = model(batch)
            loss = _batch_loss(
                outputs,
                batch,
                mode,
                class_masks,
                classification_loss,
                regression_loss,
            )
            if training:
                loss.backward()
                optimiser.step()
            row_count = batch.num_graphs
            total_loss += float(loss.detach().cpu()) * row_count
            total_rows += row_count
            if mode in {"classification", "both"}:
                for target_index, mask in enumerate(class_masks):
                    accuracy, f1 = _classification_metrics(
                        outputs[0][:, mask], batch.z[:, target_index]
                    )
                    accuracy_values[target_index].append(accuracy)
                    f1_values[target_index].append(f1)
            if mode in {"regression", "both"}:
                output_index = 0 if mode == "regression" else 1
                if mae_values is None:
                    mae_values = [[] for _ in range(outputs[output_index].shape[1])]
                for target_index in range(outputs[output_index].shape[1]):
                    observed = ~torch.isnan(batch.y[:, target_index])
                    if observed.any():
                        error = torch.abs(
                            outputs[output_index][observed, target_index]
                            - batch.y[observed, target_index]
                        ).mean()
                        mae_values[target_index].append(float(error.detach().cpu()))
    if training and scheduler is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scheduler.step()
    return EpochMetrics(
        loss=total_loss / max(total_rows, 1),
        accuracy=[float(np.mean(values)) for values in accuracy_values],
        f1=[float(np.mean(values)) for values in f1_values],
        mae=[float(np.mean(values)) if values else np.nan for values in (mae_values or [])],
    )


@torch.no_grad()
def predict_gnn_loader(
    model: GNN,
    loader: DataLoader,
    *,
    mode: str,
    phase_targets: Sequence[str] = (),
    temperature_targets: Sequence[str] = (),
    class_masks: Sequence[Sequence[bool]] = (),
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    """Return truth, predictions, confidence and absolute errors for a loader."""
    model.eval()
    records: dict[str, list[np.ndarray]] = {}

    def append(name: str, values) -> None:
        records.setdefault(name, []).append(np.atleast_1d(values))

    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch)
        if mode in {"classification", "both"}:
            for target_index, (target, mask) in enumerate(zip(phase_targets, class_masks)):
                probability = outputs[0][:, mask].softmax(dim=1)
                prediction = probability.argmax(dim=1)
                append(f"pred_{target}", prediction.cpu().numpy())
                append(f"conf_{target}", probability.max(dim=1).values.cpu().numpy())
                append(f"pos_conf_{target}", probability[:, 1:].sum(dim=1).cpu().numpy())
                if batch.z.shape[1] > target_index:
                    append(f"true_{target}", batch.z[:, target_index].cpu().numpy())
        if mode in {"regression", "both"}:
            output_index = 0 if mode == "regression" else 1
            for target_index, target in enumerate(temperature_targets):
                prediction = outputs[output_index][:, target_index]
                append(f"pred_{target}", prediction.cpu().numpy())
                if batch.y.shape[1] > target_index:
                    truth = batch.y[:, target_index]
                    append(f"true_{target}", truth.cpu().numpy())
                    append(f"error_{target}", torch.abs(truth - prediction).cpu().numpy())
    return pd.DataFrame(
        {name: np.concatenate(values) for name, values in records.items()}
    )


@dataclass(frozen=True)
class GNNTrainingConfig:
    convolution: str = "GATv2Conv"
    pooling: str = "mean-max"
    residual: bool = False
    laplacian_pe_dimension: int = 0
    hidden_channels: int = 256
    num_heads: int = 4
    num_layers: int = 12
    dropout: float = 0.5
    batch_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 300
    warmup_epochs: int = 10
    weight_decay: float = 1e-5
    id_embedding_dimension: int = 0


@dataclass
class GNNTrainingResult:
    models: list[GNN]
    predictions: pd.DataFrame
    histories: list[dict[str, list[EpochMetrics]]]


def train_gnn_cross_validation(
    dataset: LiquidCrystalGraphDataset,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    phase_targets: Sequence[str],
    temperature_targets: Sequence[str],
    mode: str = "both",
    config: GNNTrainingConfig | None = None,
    device: str | torch.device | None = None,
    checkpoint_directory: str | Path | None = None,
) -> GNNTrainingResult:
    """Train GNN folds and assemble out-of-fold predictions."""
    if mode not in {"classification", "regression", "both"}:
        raise ValueError("mode must be 'classification', 'regression' or 'both'.")
    config = config or GNNTrainingConfig()
    dataset_laplacian_pe_dimension = getattr(
        dataset, "laplacian_pe_dimension", 0
    )
    if config.laplacian_pe_dimension != dataset_laplacian_pe_dimension:
        raise ValueError(
            "The training configuration and dataset must use the same "
            "laplacian_pe_dimension."
        )
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    models = []
    histories = []
    fold_predictions = []
    output_channels = []
    if mode in {"classification", "both"}:
        output_channels.append(sum(dataset.class_counts))
    if mode in {"regression", "both"}:
        output_channels.append(len(temperature_targets))

    for fold, (train_indices, test_indices) in enumerate(splits):
        model = GNN(
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers,
            out_channels=output_channels,
            dim_node_attr=dataset.dim_node_attr + config.id_embedding_dimension,
            dim_edge_attr=dataset.dim_edge_attr,
            dropout=config.dropout,
            id_emb_dim=config.id_embedding_dimension,
            num_heads=config.num_heads,
            conv_type=config.convolution,
            pooling=config.pooling,
            residual=config.residual,
            laplacian_pe_dimension=config.laplacian_pe_dimension,
        ).to(device)
        optimiser = torch.optim.NAdam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        schedulers = [
            torch.optim.lr_scheduler.LinearLR(
                optimiser,
                start_factor=0.01,
                total_iters=max(config.warmup_epochs, 1),
            ),
            torch.optim.lr_scheduler.MultiStepLR(
                optimiser,
                milestones=[
                    max(config.epochs // 3 - config.warmup_epochs, 1),
                    max(2 * config.epochs // 3 - config.warmup_epochs, 2),
                ],
                gamma=0.1,
            ),
        ]
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimiser,
            schedulers=schedulers,
            milestones=[max(config.warmup_epochs, 1)],
        )
        training_loader = DataLoader(
            dataset[train_indices.tolist()], batch_size=config.batch_size, shuffle=True
        )
        validation_loader = DataLoader(
            dataset[test_indices.tolist()], batch_size=config.batch_size
        )
        history = {"training": [], "validation": []}
        for _ in range(config.epochs):
            history["training"].append(
                run_gnn_epoch(
                    model,
                    training_loader,
                    mode=mode,
                    class_masks=dataset.output_channel_masks,
                    optimiser=optimiser,
                    scheduler=scheduler,
                    device=device,
                )
            )
            history["validation"].append(
                run_gnn_epoch(
                    model,
                    validation_loader,
                    mode=mode,
                    class_masks=dataset.output_channel_masks,
                    device=device,
                )
            )
        prediction = predict_gnn_loader(
            model,
            validation_loader,
            mode=mode,
            phase_targets=phase_targets,
            temperature_targets=temperature_targets,
            class_masks=dataset.output_channel_masks,
            device=device,
        )
        prediction["row_index"] = test_indices
        fold_predictions.append(prediction)
        histories.append(history)
        models.append(model)
        if checkpoint_directory is not None:
            save_gnn(
                model,
                Path(checkpoint_directory) / f"{config.convolution}_cv{fold}.pt",
                metadata={"fold": fold, "training_config": asdict(config)},
            )
    predictions = (
        pd.concat(fold_predictions, ignore_index=True)
        .set_index("row_index")
        .sort_index()
    )
    predictions.insert(
        0, "SMILES", [dataset.smiles[index] for index in predictions.index]
    )
    required_phases = {f"{phase}type" for phase in CHIRAL_PHASE_TYPES}
    if (
        mode == "both"
        and set(TEMPERATURE_TARGETS).issubset(temperature_targets)
        and required_phases.issubset(phase_targets)
    ):
        predictions.insert(
            1,
            "predicted_phase_sequence",
            [format_phase_sequence(row) for row in predictions.to_dict(orient="records")],
        )
    return GNNTrainingResult(models, predictions, histories)


def predict_gnn_smiles(
    smiles: Sequence[str],
    checkpoint: str | Path,
    *,
    phase_targets: Sequence[str] = tuple(f"{phase}type" for phase in CHIRAL_PHASE_TYPES),
    temperature_targets: Sequence[str] = TEMPERATURE_TARGETS,
    atom_encoding: str = "none",
    exclude: Sequence[str] = (),
    explicit_hydrogens: bool = False,
    batch_size: int = 128,
    device: str | torch.device | None = None,
) -> pd.DataFrame:
    """Run a distributed or user-trained joint GNN checkpoint from SMILES."""
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_gnn(checkpoint, map_location=device).to(device)
    classification = pd.DataFrame(
        np.zeros((len(smiles), len(phase_targets)), dtype=int),
        columns=list(phase_targets),
    )
    regression = pd.DataFrame(
        np.zeros((len(smiles), len(temperature_targets)), dtype=float),
        columns=list(temperature_targets),
    )
    dataset = LiquidCrystalGraphDataset(
        smiles,
        classification_targets=classification,
        regression_targets=regression,
        atom_encoding=atom_encoding,
        exclude=exclude,
        explicit_hydrogens=explicit_hydrogens,
        laplacian_pe_dimension=getattr(model, "laplacian_pe_dimension", 0),
    )
    predictions = predict_gnn_loader(
        model,
        DataLoader(dataset, batch_size=batch_size),
        mode="both",
        phase_targets=phase_targets,
        temperature_targets=temperature_targets,
        class_masks=dataset.output_channel_masks,
        device=device,
    )
    predictions.insert(0, "SMILES", list(smiles))
    required_phases = {f"{phase}type" for phase in CHIRAL_PHASE_TYPES}
    if set(TEMPERATURE_TARGETS).issubset(
        temperature_targets
    ) and required_phases.issubset(phase_targets):
        predictions.insert(
            1,
            "predicted_phase_sequence",
            [
                format_phase_sequence(row)
                for row in predictions.to_dict(orient="records")
            ],
        )
    predictions.index.name = "ID"
    return predictions
