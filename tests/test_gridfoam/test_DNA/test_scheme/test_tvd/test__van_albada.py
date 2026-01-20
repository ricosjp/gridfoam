"""Tests for Van Albada TVD scheme."""

import torch

from gridfoam.DNA.scheme.tvd._van_albada import VanAlbada


def test_van_albada_correction_term():
    """Test Van Albada correction term."""
    scheme = VanAlbada()
    delta_minus = torch.tensor([1.0, 2.0])
    delta_plus = torch.tensor([2.0, 3.0])
    result = scheme.correction_term(delta_minus, delta_plus)
    assert result.shape == delta_minus.shape
    assert not torch.allclose(result, torch.zeros_like(result))


def test_van_albada_correction_term_opposite_sign():
    """Test Van Albada correction term with opposite sign deltas."""
    scheme = VanAlbada()
    delta_minus = torch.tensor([1.0, -2.0])
    delta_plus = torch.tensor([-2.0, 3.0])
    result = scheme.correction_term(delta_minus, delta_plus)
    # When signs are opposite, result should be zero
    expected = torch.zeros_like(delta_minus)
    torch.testing.assert_close(result, expected)


def test_van_albada_correction_term_shape():
    """Test Van Albada correction term preserves shape."""
    scheme = VanAlbada()
    delta_minus = torch.randn(3, 4, 5)
    delta_plus = torch.randn(3, 4, 5)
    result = scheme.correction_term(delta_minus, delta_plus)
    assert result.shape == (3, 4, 5)
