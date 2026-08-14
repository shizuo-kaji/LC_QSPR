from lc_qspr.phases import format_phase_sequence


def test_format_phase_sequence_orders_transitions():
    row = {
        "pred_Melting": 76.8,
        "pred_Clearing": 119.8,
        "pred_Atype": 1,
        "pred_Am": 76.8,
        "pred_Ap": 112.5,
        "pred_Ntype": 0,
        "pred_N*type": 1,
        "pred_Nm": 112.5,
        "pred_Np": 119.8,
    }

    assert format_phase_sequence(row) == "Cr 76.80 76.80 A 112.50 112.50 N* 119.80 119.80 is"

