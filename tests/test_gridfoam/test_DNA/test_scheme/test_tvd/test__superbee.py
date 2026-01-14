"""Tests for Superbee TVD scheme."""

import pytest
import torch

from gridfoam.DNA.scheme.tvd._superbee import Superbee


def test_superbee_correction_term():
    """Test Superbee correction term."""
    scheme = Superbee()
    delta_minus = torch.tensor([1.0, 2.0, 3.0, 4.0])
    delta_plus = torch.tensor([2.0, 3.0, 4.0, 5.0])
    result = scheme.correction_term(delta_minus, delta_plus)
    # Verify shape and that result is not all zeros
    assert result.shape == delta_minus.shape
    assert not torch.allclose(result, torch.zeros_like(result))


def test_superbee_correction_term_opposite_sign():
    """Test Superbee correction term with opposite sign deltas."""
    scheme = Superbee()
    delta_minus = torch.tensor([1.0, -2.0])
    delta_plus = torch.tensor([-2.0, 3.0])
    result = scheme.correction_term(delta_minus, delta_plus)
    # When signs are opposite, result should be zero
    expected = torch.zeros_like(delta_minus)
    torch.testing.assert_close(result, expected)


def test_superbee_correction_term_shape():
    """Test Superbee correction term preserves shape."""
    scheme = Superbee()
    delta_minus = torch.randn(3, 4, 5)
    delta_plus = torch.randn(3, 4, 5)
    result = scheme.correction_term(delta_minus, delta_plus)
    assert result.shape == (3, 4, 5)
