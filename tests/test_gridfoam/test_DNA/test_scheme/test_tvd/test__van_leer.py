"""Tests for Van Leer TVD scheme."""

import torch

from gridfoam.DNA.scheme.tvd._van_leer import VanLeer


def test_van_leer_correction_term():
    """Test Van Leer correction term."""
    scheme = VanLeer()
    delta_minus = torch.tensor([1.0, 2.0])
    delta_plus = torch.tensor([2.0, 3.0])
    result = scheme.correction_term(delta_minus, delta_plus)
    # f = [1/3, 2/5], phi = 4*f*(1-f) = [8/9, 24/25]
    # correction = 0.25 * phi * (delta_minus + delta_plus)
    # = 0.25 * [8/9, 24/25] * [3, 5] = [2/3, 24/25]
    assert result.shape == delta_minus.shape
    assert not torch.allclose(result, torch.zeros_like(result))


def test_van_leer_correction_term_opposite_sign():
    """Test Van Leer correction term with opposite sign deltas."""
    scheme = VanLeer()
    delta_minus = torch.tensor([1.0, -2.0])
    delta_plus = torch.tensor([-2.0, 3.0])
    result = scheme.correction_term(delta_minus, delta_plus)
    # When signs are opposite, result should be zero
    expected = torch.zeros_like(delta_minus)
    torch.testing.assert_close(result, expected)


def test_van_leer_correction_term_shape():
    """Test Van Leer correction term preserves shape."""
    scheme = VanLeer()
    delta_minus = torch.randn(3, 4, 5)
    delta_plus = torch.randn(3, 4, 5)
    result = scheme.correction_term(delta_minus, delta_plus)
    assert result.shape == (3, 4, 5)
