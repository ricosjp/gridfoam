"""Tests for harmonic_mean utility."""

import pytest
import torch

from gridfoam.DNA.utils.harmonic_mean import harmonic_mean


def test_harmonic_mean_basic():
    """Test basic harmonic mean calculation."""
    forward = torch.tensor([1.0, 2.0, 3.0])
    backward = torch.tensor([2.0, 3.0, 4.0])
    result = harmonic_mean(forward, backward)
    expected = torch.tensor([4.0 / 3.0, 12.0 / 5.0, 24.0 / 7.0])
    torch.testing.assert_close(result, expected)


def test_harmonic_mean_zero_denominator():
    """Test harmonic mean with zero denominator."""
    forward = torch.tensor([0.0, 1.0])
    backward = torch.tensor([0.0, 2.0])
    result = harmonic_mean(forward, backward)
    assert result[0] == 0.0
    assert result[1] == pytest.approx(4.0 / 3.0)


def test_harmonic_mean_shape():
    """Test harmonic mean preserves shape."""
    forward = torch.randn(3, 4, 5)
    backward = torch.randn(3, 4, 5)
    result = harmonic_mean(forward, backward)
    assert result.shape == (3, 4, 5)


def test_harmonic_mean_dtype():
    """Test harmonic mean preserves dtype."""
    forward = torch.tensor([1.0], dtype=torch.float32)
    backward = torch.tensor([2.0], dtype=torch.float32)
    result = harmonic_mean(forward, backward)
    assert result.dtype == torch.float32
