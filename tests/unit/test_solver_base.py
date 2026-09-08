"""
Residual thresholds use the larger absolute or relative bound for strict
stopping.
"""

from __future__ import annotations

import torch

from gridfoam.solvers.base import residual_threshold


def test_residual_threshold_combines_atol_rtol() -> None:
    """OR criteria use the larger threshold, not their sum."""
    atol, rtol = 1e-8, 1e-6
    ref = torch.tensor(10.0)
    out = residual_threshold(atol, rtol, ref)
    expected = rtol * ref
    assert torch.allclose(out, expected)


def test_residual_between_max_and_sum_does_not_converge() -> None:
    """
    The threshold supports strict max-based stopping, including equality
    and near misses.
    """
    threshold = residual_threshold(0.1, 0.1, torch.tensor(1.0))
    # Convergence is strict: residual == max(atol, rtol * ref) is not enough.
    assert not (torch.tensor(0.15) < threshold).item()
    assert not (torch.tensor(0.1) < threshold).item()
    assert (torch.tensor(0.09) < threshold).item()
