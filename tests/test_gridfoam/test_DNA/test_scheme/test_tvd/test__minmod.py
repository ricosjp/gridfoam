"""Tests for Minmod TVD scheme."""

import pytest
import torch

from gridfoam.DNA.scheme.tvd._minmod import Minmod


def test_minmod_correction_term_same_sign():
    """Test Minmod correction term with same sign deltas."""
    scheme = Minmod()
    delta_minus = torch.tensor([1.0, 2.0])
    delta_plus = torch.tensor([2.0, 3.0])
    result = scheme.correction_term(delta_minus, delta_plus)
    # f = [1/3, 2/5], phi = min(2f, 2(1-f)) = [2/3, 4/5]
    # correction = 0.25 * phi * (delta_minus + delta_plus)
    # = 0.25 * [2/3, 4/5] * [3, 5] = [0.5, 1.0]
    expected = torch.tensor([0.5, 1.0])
    torch.testing.assert_close(result, expected, rtol=1e-5)


def test_minmod_correction_term_opposite_sign():
    """Test Minmod correction term with opposite sign deltas."""
    scheme = Minmod()
    delta_minus = torch.tensor([1.0, -2.0])
    delta_plus = torch.tensor([-2.0, 3.0])
    result = scheme.correction_term(delta_minus, delta_plus)
    # When signs are opposite, mask is False, so f remains 0
    # phi = min(2*0, 2*(1-0)) = 0
    expected = torch.zeros_like(delta_minus)
    torch.testing.assert_close(result, expected)


def test_minmod_correction_term_shape():
    """Test Minmod correction term preserves shape."""
    scheme = Minmod()
    delta_minus = torch.randn(3, 4, 5)
    delta_plus = torch.randn(3, 4, 5)
    result = scheme.correction_term(delta_minus, delta_plus)
    assert result.shape == (3, 4, 5)
