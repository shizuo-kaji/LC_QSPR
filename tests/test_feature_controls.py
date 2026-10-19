from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lc_qspr.feature_controls import ScalarCodePermutation  # noqa: E402


def test_fixed_permutation_preserves_codes_and_global_random_state():
    values = torch.tensor([0.0, 0.5, 0.5, 1.0, 2.0, 3.0])
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = np.random.get_state()
    mapping = ScalarCodePermutation.fit(values, seed=7)
    assert torch.equal(torch_before, torch.random.get_rng_state())
    np.testing.assert_array_equal(numpy_before[1], np.random.get_state()[1])
    assert mapping == ScalarCodePermutation.fit(values.flip(0), seed=7)
    result = mapping.apply(values)
    assert torch.equal(values[:, None] == values, result[:, None] == result)
    assert not torch.equal(values, result)
    assert torch.equal(torch.unique(values), torch.unique(result))
    assert torch.equal(mapping.apply(values[:3]), result[:3])
    assert torch.equal(values, torch.tensor([0.0, 0.5, 0.5, 1.0, 2.0, 3.0]))
    assert ScalarCodePermutation(**asdict(mapping)) == mapping


def test_permutation_rejects_unknown_nonfinite_and_invalid_codes():
    mapping = ScalarCodePermutation.fit(torch.tensor([0.0, 1.0, 2.0]), seed=13)
    for values in (torch.tensor([-1.0]), torch.tensor([0.5]), torch.tensor([3.0])):
        with pytest.raises(ValueError, match="Unseen"):
            mapping.apply(values)
    for values in (torch.tensor([]), torch.tensor([float("nan")]), torch.tensor([1])):
        with pytest.raises(ValueError):
            ScalarCodePermutation.fit(values, seed=7)
    with pytest.raises(ValueError):
        mapping.apply(torch.tensor([float("inf")]))
    with pytest.raises(ValueError):
        ScalarCodePermutation((0.0, 1.0), (1.0, 1.0))


def test_permutation_preserves_shape_and_dtype():
    values = torch.tensor([[0.0, 1.0], [2.0, 0.0]], dtype=torch.float64)
    mapping = ScalarCodePermutation.fit(values, seed=29)
    result = mapping.apply(values)
    assert result.dtype == values.dtype
    assert result.shape == values.shape
    assert result.device == values.device
    assert mapping.apply(values[:0]).shape == (0, 2)
