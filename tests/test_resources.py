from lc_qspr import descriptor_catalogue, model_descriptor_names


def test_packaged_tables_are_available():
    catalogue = descriptor_catalogue()
    names = model_descriptor_names()

    assert {"name", "Shape", "Electric", "Flexible"}.issubset(catalogue.columns)
    assert len(names) == 1613
    assert len(names) == len(set(names))

