import numpy as np
import pandas as pd

from lc_qspr.catboost_training import (
    CatBoostConfig,
    train_catboost_phases,
    train_catboost_temperatures,
)
from lc_qspr.evaluation import cross_validation_splits
from lc_qspr.lightgbm_training import (
    LightGBMConfig,
    train_lightgbm_phases,
    train_lightgbm_temperatures,
)


def _training_table():
    random = np.random.default_rng(42)
    table = pd.DataFrame(
        {
            "SMILES": ["CC"] * 20,
            "f1": random.normal(size=20),
            "f2": random.normal(size=20),
            "Ntype": [0, 1] * 10,
            "Atype": [0, 0, 1, 1] * 5,
        }
    )
    table["Melting"] = 20 + 2 * table["f1"]
    table["Clearing"] = 30 + 2 * table["f2"]
    return table


def test_catboost_and_lightgbm_training_smoke_paths():
    table = _training_table()
    splits = cross_validation_splits(
        table, ["Ntype", "Atype"], mode="classification", n_splits=2
    )
    cat_phase = train_catboost_phases(
        table,
        ["f1", "f2"],
        ["Ntype", "Atype"],
        splits,
        config=CatBoostConfig(iterations=2, verbose=False),
    )
    cat_temperature = train_catboost_temperatures(
        table,
        ["f1", "f2"],
        ["Melting", "Clearing"],
        splits,
        config=CatBoostConfig(iterations=2, verbose=False),
    )
    lgb_phase = train_lightgbm_phases(
        table,
        ["f1", "f2"],
        ["Ntype"],
        splits,
        config=LightGBMConfig(iterations=2, parameters={"min_data_in_leaf": 1}),
    )
    lgb_temperature = train_lightgbm_temperatures(
        table,
        ["f1", "f2"],
        ["Melting"],
        splits,
        config=LightGBMConfig(iterations=2, parameters={"min_data_in_leaf": 1}),
    )

    assert len(cat_phase.models) == 2
    assert cat_temperature.predictions["pred_Melting"].notna().all()
    assert len(lgb_phase.models["Ntype"]) == 2
    assert lgb_temperature.predictions["pred_Melting"].notna().all()

