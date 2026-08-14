import pytest

from lc_qspr import calculate_descriptors, model_descriptor_names


def test_calculate_descriptors_for_valid_smiles():
    table = calculate_descriptors(["CCO"])
    assert table.shape[0] == 1
    assert set(model_descriptor_names()).issubset(table.columns)


def test_invalid_smiles_reports_row_number():
    with pytest.raises(ValueError, match="row\\(s\\): 1"):
        calculate_descriptors(["CCO", "not-a-smiles"])

