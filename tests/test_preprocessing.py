import pandas as pd

from lc_qspr.preprocessing import (
    normalise_legacy_phase_columns,
    parse_phase_table,
    separate_chiral_phases,
)


def test_parse_phase_table_extracts_classes_and_temperatures():
    source = pd.DataFrame(
        {
            "ID": [7],
            "SMILES": ["CC"],
            "Phases": ["Cr 76.8 A 112.5 N* 119.8 is"],
        }
    ).set_index("ID")

    parsed, report = parse_phase_table(source)

    assert report.excluded == 0
    assert parsed.loc[7, "Melting"] == 76.8
    assert parsed.loc[7, "Atype"] == 1
    assert parsed.loc[7, "Ntype"] == 2
    assert parsed.loc[7, "Np"] == 119.8
    assert parsed.loc[7, "Clearing"] == 119.8


def test_legacy_d_columns_and_chiral_classes_are_normalised():
    table = pd.DataFrame(
        {"Dtype": [0], "Dm": [1.0], "Dp": [2.0], "Ntype": [2], "Nm": [3.0], "Np": [4.0]}
    )
    normalised = normalise_legacy_phase_columns(table)
    separated, phases = separate_chiral_phases(normalised)

    assert {"Dhtype", "Dhm", "Dhp"}.issubset(separated.columns)
    assert phases == ["N*"]
    assert separated.loc[0, "Ntype"] == 0
    assert separated.loc[0, "N*type"] == 1
    assert separated.loc[0, "N*m"] == 3.0

