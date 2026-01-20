"""Tests for FVC div linear operator."""

import torch

from gridfoam.DNA.fielddata import CellField
from gridfoam.DNA.scheme.fvc.div._linear import FVCDivLinear


def test_fvc_div_linear_apply():
    """Test FVCDivLinear apply method."""
    # Create a simple cell field
    T, C, N, H = 1, 3, 4, 2
    device = torch.device("cpu")
    dtype = torch.float32
    cell_field = CellField(T, C, N, H, dtype, device)
    cell_field.raw = torch.ones(T, C, N+2*H, N+2*H, N+2*H)

    dx = torch.tensor([1.0, 1.0, 1.0])
    result = FVCDivLinear.apply(cell_field, dx)
    assert result.shape == (T, 1, N, N, N)
    # For uniform field, divergence should be zero
    torch.testing.assert_close(result[0], torch.zeros(1, N, N, N), rtol=1e-5, atol=1e-6)


def test_fvc_div_linear_apply_shape():
    """Test FVCDivLinear apply preserves expected shape."""
    T, C, N, H = 2, 3, 5, 2
    device = torch.device("cpu")
    dtype = torch.float32
    cell_field = CellField(T, C, N, H, dtype, device)
    cell_field.raw = torch.randn(T, C, N+2*H, N+2*H, N+2*H)

    dx = torch.tensor([0.5, 0.5, 0.5])
    result = FVCDivLinear.apply(cell_field, dx)
    assert result.shape == (T, 1, N, N, N)


def test_fvc_div_linear_apply_linear_field_exact():
    """Test FVCDivLinear apply with linear field - exact values."""
    T, C, N, H = 1, 3, 4, 2
    device = torch.device("cpu")
    dtype = torch.float32
    cell_field = CellField(T, C, N, H, dtype, device)

    # Create linear field: psi[i] = i
    for i in range(N+2*H):
        cell_field.raw[0, 0, :, :, i] = float(i)

    dx = torch.tensor([1.0, 1.0, 1.0])
    result = FVCDivLinear.apply(cell_field, dx)

    expected = torch.ones(1, N, N, N)
    torch.testing.assert_close(
        result[0, :, :, :, :],
        expected,
    )


def test_fvc_div_linear_apply_linear_field_3d():
    """Test FVCDivLinear apply with 3D linear field."""
    # Linear field: psi = (x, y, z)
    # Divergence should be 3.0 (1.0 in each direction)

    T, C, N, H = 1, 3, 4, 2
    device = torch.device("cpu")
    dtype = torch.float32
    cell_field = CellField(T, C, N, H, dtype, device)

    # Create 3D linear field: psi[i,j,k] = (i, j, k)
    for k in range(N+2*H):
        for j in range(N+2*H):
            for i in range(N+2*H):
                cell_field.raw[0, 0, k, j, i] = float(i)
                cell_field.raw[0, 1, k, j, i] = float(j)
                cell_field.raw[0, 2, k, j, i] = float(k)

    dx = torch.tensor([1.0, 1.0, 1.0])
    result = FVCDivLinear.apply(cell_field, dx)

    # Expected: divergence = 3.0 (1.0 in each direction)
    expected = torch.full((1, N, N, N), 3.0)
    torch.testing.assert_close(
        result[0],
        expected,
    )
