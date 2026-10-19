import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from rdkit import Chem  # noqa: E402

from lc_qspr import gnn  # noqa: E402
from lc_qspr.gnn import (  # noqa: E402
    GNN,
    GNNTrainingConfig,
    LiquidCrystalGraphDataset,
    load_gnn,
    predict_gnn_smiles,
    save_gnn,
    smiles_to_graph,
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
            pooling="mean-max-size",
            residual=True,
            batch_size=2,
            epochs=1,
            warmup_epochs=1,
        ),
        device="cpu",
    )

    assert len(result.models) == 1
    assert result.models[0].pooling == "mean-max-size"
    assert result.models[0].residual is True
    assert result.predictions.index.tolist() == [0, 1]
    assert np.isfinite(result.histories[0]["validation"][0].loss)


def test_sum_pooling_and_residual_connections_round_trip(tmp_path):
    dataset = LiquidCrystalGraphDataset(
        ["CC", "CCCC"],
        classification_targets=pd.DataFrame({"Ntype": [0, 1]}),
        regression_targets=pd.DataFrame({"Melting": [1.0, 2.0]}),
        atom_encoding="none",
        laplacian_pe_dimension=2,
    )
    model = GNN(
        hidden_channels=8,
        num_layers=2,
        out_channels=(2, 1),
        dim_node_attr=dataset.dim_node_attr,
        dim_edge_attr=dataset.dim_edge_attr,
        num_heads=2,
        pooling="mean-max-sum",
        residual=True,
        laplacian_pe_dimension=2,
    )
    checkpoint = tmp_path / "gnn.pt"

    save_gnn(model, checkpoint)
    restored = load_gnn(checkpoint, map_location="cpu")

    assert restored.pooling == "mean-max-sum"
    assert restored.residual is True
    assert restored.laplacian_pe_dimension == 2
    assert restored.convs[0].in_channels == dataset.dim_node_attr + 2
    assert restored.fc1[0].in_features == 24


def test_gps_conv_training_and_portable_checkpoint(tmp_path):
    dataset = LiquidCrystalGraphDataset(
        ["CC", "CCC", "CCO", "CCN", "CCCl", "CCBr"],
        classification_targets=pd.DataFrame({"Ntype": [0, 1, 0, 1, 0, 1]}),
        regression_targets=pd.DataFrame(
            {"Melting": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
        ),
        laplacian_pe_dimension=3,
    )
    result = train_gnn_cross_validation(
        dataset,
        [(np.array([2, 3, 4, 5]), np.array([0, 1]))],
        phase_targets=["Ntype"],
        temperature_targets=["Melting"],
        config=GNNTrainingConfig(
            convolution="GPSConv",
            laplacian_pe_dimension=3,
            hidden_channels=8,
            num_heads=2,
            num_layers=2,
            pooling="mean-max-sum",
            batch_size=2,
            epochs=1,
            warmup_epochs=1,
        ),
        device="cpu",
    )
    model = result.models[0]
    checkpoint = tmp_path / "gps.pt"

    assert model.conv_type == "GPSConv"
    assert type(model.convs[0]).__name__ == "GPSConv"
    assert type(model.convs[0].conv).__name__ == "GINEConv"
    assert np.isfinite(result.histories[0]["validation"][0].loss)

    save_gnn(model, checkpoint)
    restored = load_gnn(checkpoint, map_location="cpu")

    assert restored.conv_type == "GPSConv"
    assert restored.hidden_channels == 8
    assert restored.num_heads == 2
    assert restored.dim_node_attr == dataset.dim_node_attr
    assert restored.dim_edge_attr == dataset.dim_edge_attr
    assert restored.laplacian_pe_dimension == 3
    assert restored.fc1[0].in_features == 24

    prediction = predict_gnn_smiles(
        ["CCO"],
        checkpoint,
        phase_targets=["Ntype"],
        temperature_targets=["Melting"],
        atom_encoding="one-hot",
        device="cpu",
    )
    assert prediction.shape[0] == 1


def test_gps_conv_requires_compatible_attention_heads():
    with pytest.raises(ValueError, match="divides hidden_channels"):
        GNN(
            hidden_channels=10,
            num_heads=4,
            num_layers=1,
            out_channels=(1,),
            dim_node_attr=3,
            dim_edge_attr=2,
            conv_type="GPSConv",
        )


def test_laplacian_pe_ignores_bond_multiplicity():
    single = smiles_to_graph("CC", laplacian_pe_dimension=3)
    double = smiles_to_graph("C=C", laplacian_pe_dimension=3)
    triple = smiles_to_graph("C#C", laplacian_pe_dimension=3)

    np.testing.assert_allclose(single.laplacian_pe, double.laplacian_pe)
    np.testing.assert_allclose(single.laplacian_pe, triple.laplacian_pe)
    assert single.laplacian_pe.shape == (2, 3)
    assert torch.count_nonzero(single.laplacian_pe[:, 1:]) == 0


def test_feature_block_registry_matches_emitted_widths():
    """The registry must describe exactly what the featurisers emit."""
    molecule = Chem.MolFromSmiles("CCCCCCc1ccc(cc1)/C=N/c1ccc(OC)cc1")
    assert gnn.FULL_NODE_FEATURE_WIDTH == 88
    assert gnn.FULL_EDGE_FEATURE_WIDTH == 10
    node_values, _ = gnn.atom_features(molecule.GetAtomWithIdx(0), exclude=())
    assert node_values.shape == (gnn.FULL_NODE_FEATURE_WIDTH,)
    edge_values = gnn.bond_features(molecule.GetBondWithIdx(0), exclude=())
    assert edge_values.shape == (gnn.FULL_EDGE_FEATURE_WIDTH,)


def test_placeholder_keeps_full_width_and_matches_drop_mode():
    """Placeholder mode must equal drop mode outside the ablated columns."""
    molecule = Chem.MolFromSmiles("CCCCCCc1ccc(cc1)/C=N/c1ccc(OC)cc1")
    ablations = (
        (),
        ("chirality_type", "stereochemistry"),
        ("n_valence", "n_hydrogens"),
        ("hybridisation_type", "is_aromatic", "bond_is_conjugated"),
        ("bond_type",),
    )
    for exclude in ablations:
        node_spans = gnn.ablated_feature_spans(gnn.NODE_FEATURE_BLOCKS, exclude)
        edge_spans = gnn.ablated_feature_spans(gnn.EDGE_FEATURE_BLOCKS, exclude)
        for atom in molecule.GetAtoms():
            full, _ = gnn.atom_features(atom, exclude=())
            placed, _ = gnn.atom_features(
                atom, exclude=exclude, placeholder_excluded=True
            )
            dropped, _ = gnn.atom_features(atom, exclude=exclude)
            keep = np.ones(gnn.FULL_NODE_FEATURE_WIDTH, dtype=bool)
            for start, stop, _kind in node_spans:
                keep[start:stop] = False
            assert placed.shape == (gnn.FULL_NODE_FEATURE_WIDTH,)
            assert np.array_equal(placed[keep], full[keep])
            assert not placed[~keep].any()
            assert np.array_equal(dropped, full[keep])
        for bond in molecule.GetBonds():
            full = gnn.bond_features(bond, exclude=())
            placed = gnn.bond_features(
                bond, exclude=exclude, placeholder_excluded=True
            )
            keep = np.ones(gnn.FULL_EDGE_FEATURE_WIDTH, dtype=bool)
            for start, stop, _kind in edge_spans:
                keep[start:stop] = False
            assert placed.shape == (gnn.FULL_EDGE_FEATURE_WIDTH,)
            assert np.array_equal(placed[keep], full[keep])
            assert not placed[~keep].any()


def test_ablated_feature_spans_cover_atom_identity_and_aliases():
    """Element identity follows atom_encoding, and aliases are honoured."""
    spans = gnn.ablated_feature_spans(
        gnn.NODE_FEATURE_BLOCKS, (), atom_encoding="none"
    )
    assert spans[0] == (0, len(gnn.PERMITTED_ATOMS), "one-hot")
    assert gnn.ablated_feature_spans(gnn.NODE_FEATURE_BLOCKS, ()) == ()
    by_alias = gnn.ablated_feature_spans(
        gnn.NODE_FEATURE_BLOCKS, ("n_heavy_neighbors",)
    )
    by_name = gnn.ablated_feature_spans(
        gnn.NODE_FEATURE_BLOCKS, ("n_heavy_neighbours",)
    )
    assert by_alias == by_name != ()


def test_randomise_ablated_slots_respects_support_and_leaves_others_intact():
    """Noise must land only in ablated columns, on the right support."""
    spans = (
        (0, 4, "one-hot"),
        (4, 5, "boolean"),
        (5, 6, "scalar"),
    )
    values = torch.zeros(64, 8)
    values[:, 6:] = 7.0
    gnn.randomise_ablated_slots(values, spans)
    one_hot_block = values[:, 0:4]
    assert torch.equal(one_hot_block.sum(dim=1), torch.ones(64))
    assert set(one_hot_block.unique().tolist()) <= {0.0, 1.0}
    assert set(values[:, 4].unique().tolist()) <= {0.0, 1.0}
    assert values[:, 5].std() > 0
    assert torch.equal(values[:, 6:], torch.full((64, 2), 7.0))


def test_randomise_ablated_slots_carries_no_molecule_information():
    """Two draws must differ, so the noise cannot encode the input."""
    spans = ((0, 6, "scalar"),)
    first = torch.zeros(128, 6)
    second = torch.zeros(128, 6)
    gnn.randomise_ablated_slots(first, spans)
    gnn.randomise_ablated_slots(second, spans)
    assert not torch.allclose(first, second)
    # A seeded generator makes evaluation reproducible.
    seeded_a, seeded_b = torch.zeros(128, 6), torch.zeros(128, 6)
    for target in (seeded_a, seeded_b):
        generator = torch.Generator()
        generator.manual_seed(11)
        gnn.randomise_ablated_slots(target, spans, generator)
    assert torch.equal(seeded_a, seeded_b)


def test_randomise_ablated_slots_breaks_node_symmetry():
    """Distinct per-node draws are what rescue an all-constant input."""
    spans = ((0, 88, "one-hot"),)
    values = torch.zeros(256, 88)
    gnn.randomise_ablated_slots(values, spans)
    assert values.unique(dim=0).shape[0] > 1


def test_dataset_placeholder_mode_keeps_registered_widths():
    """A fully ablated dataset still presents 88 node and 10 edge columns."""
    every_attribute = tuple(
        block.name for block in gnn.NODE_FEATURE_BLOCKS if block.name != "atom_identity"
    ) + tuple(block.name for block in gnn.EDGE_FEATURE_BLOCKS)
    dataset = gnn.LiquidCrystalGraphDataset(
        ["CCO", "c1ccccc1", "CCCCCCc1ccc(cc1)C#N"],
        atom_encoding="none",
        exclude=every_attribute,
        placeholder_excluded=True,
    )
    graph = dataset[0]
    assert graph.x.shape[1] == gnn.FULL_NODE_FEATURE_WIDTH
    assert graph.edge_attr.shape[1] == gnn.FULL_EDGE_FEATURE_WIDTH
    assert not graph.x.any() and not graph.edge_attr.any()
