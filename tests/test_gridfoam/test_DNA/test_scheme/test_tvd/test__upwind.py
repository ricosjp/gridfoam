"""Tests for Upwind TVD scheme."""

import torch

from gridfoam.DNA.scheme.tvd._upwind import Upwind


def test_upwind_correction_term():
    """Test Upwind correction term returns zeros."""
    scheme = Upwind()
    delta_minus = torch.tensor([1.0, 2.0, 3.0])
    delta_plus = torch.tensor([2.0, 3.0, 4.0])
    result = scheme.correction_term(delta_minus, delta_plus)
    expected = torch.zeros_like(delta_minus)
    torch.testing.assert_close(result, expected)


def test_upwind_correction_term_shape():
    """Test Upwind correction term preserves shape."""
    scheme = Upwind()
    delta_minus = torch.randn(3, 4, 5)
    delta_plus = torch.randn(3, 4, 5)
    result = scheme.correction_term(delta_minus, delta_plus)
    assert result.shape == (3, 4, 5)
