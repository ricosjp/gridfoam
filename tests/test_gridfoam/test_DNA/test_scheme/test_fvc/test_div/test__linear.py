"""Tests for FVC div linear operator."""

import pytest
import torch

from gridfoam.DNA.fielddata import CellField
from gridfoam.DNA.scheme.fvc.div._linear import FVCDivLinear


def test_fvc_div_linear_apply():
    """Test FVCDivLinear apply method."""
    # Create a simple cell field
    T, C, N = 1, 1, 4
    device = torch.device("cpu")
    dtype = torch.float32
    cell_field = CellField(T, C, N, 1, dtype, device)  # H=1
    cell_field.interior[0] = torch.ones(C, N, N, N)

    dx = torch.tensor([1.0, 1.0, 1.0])
    result = FVCDivLinear.apply(cell_field, dx)
    assert result.shape == (T, 1, N, N, N)
    # For uniform field, divergence should be zero
    torch.testing.assert_close(result[0], torch.zeros(1, N, N, N), rtol=1e-5, atol=1e-6)


def test_fvc_div_linear_apply_shape():
    """Test FVCDivLinear apply preserves expected shape."""
    T, C, N = 2, 3, 5
    device = torch.device("cpu")
    dtype = torch.float32
    cell_field = CellField(T, C, N, 1, dtype, device)  # H=1
    cell_field.interior[0] = torch.randn(C, N, N, N)

    dx = torch.tensor([0.5, 0.5, 0.5])
    result = FVCDivLinear.apply(cell_field, dx)
    assert result.shape == (T, 1, N, N, N)


def test_fvc_div_linear_apply_linear_field_exact():
    """Test FVCDivLinear apply with linear field - exact values."""
    # Linear field: psi = x (in x direction)
    # For cell i at position i*dx, value is i
    # Face between cell i and i+1: face_average = (i + i+1)/2 = i + 0.5
    # Divergence = (face_value[i+1] - face_value[i]) / dx
    # For uniform dx=1.0, divergence should be 1.0 everywhere
    
    T, C, N = 1, 1, 4
    device = torch.device("cpu")
    dtype = torch.float32
    cell_field = CellField(T, C, N, 1, dtype, device)  # H=1
    
    # Create linear field: psi[i] = i
    for i in range(N):
        cell_field.interior[0, 0, :, :, i] = float(i)

    dx = torch.tensor([1.0, 1.0, 1.0])
    result = FVCDivLinear.apply(cell_field, dx)
    
    # Expected: divergence = 1.0 in x direction (constant)
    # Note: internal cells only (boundary may differ due to H=1)
    # For N=4, internal cells are indices 0, 1, 2, 3
    # Divergence should be approximately 1.0 for internal cells
    expected = torch.ones(1, N, N, N)
    # Allow some tolerance due to face averaging at boundaries
    torch.testing.assert_close(
        result[0, :, :, :, :], 
        expected, 
        rtol=1e-3, 
        atol=1e-3
    )


def test_fvc_div_linear_apply_linear_field_3d():
    """Test FVCDivLinear apply with 3D linear field."""
    # Linear field: psi = x + y + z
    # Divergence should be 3.0 (1.0 in each direction)
    
    T, C, N = 1, 1, 3
    device = torch.device("cpu")
    dtype = torch.float32
    cell_field = CellField(T, C, N, 1, dtype, device)  # H=1
    
    # Create 3D linear field: psi[i,j,k] = i + j + k
    for i in range(N):
        for j in range(N):
            for k in range(N):
                cell_field.interior[0, 0, i, j, k] = float(i + j + k)

    dx = torch.tensor([1.0, 1.0, 1.0])
    result = FVCDivLinear.apply(cell_field, dx)
    
    # Expected: divergence = 3.0 (1.0 in each direction)
    expected = torch.full((1, N, N, N), 3.0)
    torch.testing.assert_close(
        result[0], 
        expected, 
        rtol=1e-2, 
        atol=1e-2
    )
