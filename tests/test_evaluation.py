import pandas as pd

from lc_qspr.evaluation import classification_summary, temperature_summary


def test_evaluation_summaries_use_saved_result_columns():
    result = pd.DataFrame(
        {
            "true_Ntype": [0, 1, 1, 0],
            "pred_Ntype": [0, 1, 0, 0],
            "pos_conf_Ntype": [0.1, 0.9, 0.4, 0.2],
            "error_Melting": [1.0, 2.0, None, 3.0],
        }
    )

    classification = classification_summary(result, ["Ntype"])
    temperature = temperature_summary(result, ["Melting"])

    assert classification.loc["N", "accuracy"] == 0.75
    assert classification.loc["N", "auc"] == 1.0
    assert temperature.loc["Melting", "n"] == 3
    assert temperature.loc["Melting", "mae"] == 2.0

