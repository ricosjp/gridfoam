"""Unit tests for ``gridfoam.solvers.base`` helpers."""

from __future__ import annotations

import torch

from gridfoam.solvers.base import is_converged, residual_threshold


def test_is_converged_true_when_below_threshold():
    # Converged when normalized residual is strictly below threshold.
    thresh = torch.tensor([1e-2])
    norm_res = torch.tensor([1e-3])
    assert is_converged(thresh, norm_res) is True


def test_is_converged_false_when_above_threshold():
    # Not converged when normalized residual exceeds threshold.
    thresh = torch.tensor([1e-4])
    norm_res = torch.tensor([1e-2])
    assert is_converged(thresh, norm_res) is False


def test_residual_threshold_combines_atol_rtol():
    # Effective tolerance is atol + rtol * reference_norm.
    atol, rtol = 1e-8, 1e-6
    ref = torch.tensor([10.0])
    out = residual_threshold(atol, rtol, ref)
    expected = atol + rtol * ref
    assert torch.allclose(out, expected)
