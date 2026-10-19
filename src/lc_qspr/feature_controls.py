"""Information-preserving numerical-code controls for feature ablations."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class ScalarCodePermutation:
    """A fixed bijection of distinct training-set scalar values.

    Equal input values remain equal and unequal values remain unequal. Thus
    the numerical geometry changes without changing the information in the
    scalar codes. No labels are used, and unseen codes are rejected rather
    than silently assigned a value. Fit this control on training nodes only.
    It is not a shuffle across individual observations.
    """

    source_values: tuple[float, ...]
    target_values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.source_values or not all(np.isfinite(self.source_values)):
            raise ValueError("Source codes must be non-empty and finite.")
        if tuple(sorted(set(self.source_values))) != self.source_values:
            raise ValueError("Source codes must be sorted and distinct.")
        if sorted(self.target_values) != list(self.source_values):
            raise ValueError("Target codes must be a permutation of source codes.")

    @classmethod
    def fit(cls, values: Tensor, *, seed: int) -> "ScalarCodePermutation":
        """Create a reproducible mapping without changing global random state."""
        if not values.is_floating_point() or not values.numel() or not torch.isfinite(values).all():
            raise ValueError("Training codes must be non-empty finite floating-point values.")
        source = tuple(torch.unique(values.detach().cpu(), sorted=True).tolist())
        order = np.random.default_rng(seed).permutation(len(source))
        return cls(source, tuple(source[i] for i in order))

    def apply(self, values: Tensor) -> Tensor:
        """Return mapped values, preserving shape, device and floating dtype."""
        if not values.is_floating_point() or not torch.isfinite(values).all():
            raise ValueError("Codes must be finite floating-point values.")
        source = torch.tensor(self.source_values, dtype=values.dtype, device=values.device)
        target = torch.tensor(self.target_values, dtype=values.dtype, device=values.device)
        if len(torch.unique(source)) != len(source):
            raise ValueError("Input dtype merges distinct source codes.")
        indices = torch.searchsorted(source, values.contiguous())
        safe = indices.clamp(max=len(source) - 1)
        if not torch.all((indices < len(source)) & (source[safe] == values)):
            raise ValueError("Unseen scalar code; fit and evaluation vocabularies differ.")
        return target[indices]
