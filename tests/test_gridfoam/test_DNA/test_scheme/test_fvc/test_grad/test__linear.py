"""Tests for FVC grad linear operator."""

import torch

from gridfoam.DNA.fielddata import CellField
from gridfoam.DNA.scheme.fvc.grad._linear import FVCGradLinear


def test_fvc_grad_linear_apply():
    """Test FVCGradLinear apply method."""
    # Create a simple cell field
    T, C, N, H = 1, 1, 4, 2
    device = torch.device("cpu")
    dtype = torch.float32
    cell_field = CellField(T, C, N, H, dtype, device)
    cell_field.raw[0] = torch.ones(T, C, N + 2 * H, N + 2 * H, N + 2 * H)

    dx = torch.tensor([1.0, 1.0, 1.0])
    result = FVCGradLinear.apply(cell_field, dx)
    # Expected shape: (T, 3, C, N, N, N)
    assert len(result.shape) == 6
    assert result.shape[0] == T
    assert result.shape[1] == 3  # gradient components
    # For uniform field, gradient should be zero
    torch.testing.assert_close(
        result, torch.zeros(T, 3, C, N, N, N), rtol=1e-5, atol=1e-6
    )


def test_fvc_grad_linear_apply_shape():
    """Test FVCGradLinear apply preserves expected shape."""
    T, C, N = 2, 3, 5
    device = torch.device("cpu")
    dtype = torch.float32
    cell_field = CellField(T, C, N, 1, dtype, device)  # H=1
    cell_field.interior[0] = torch.randn(C, N, N, N)

    dx = torch.tensor([0.5, 0.5, 0.5])
    result = FVCGradLinear.apply(cell_field, dx)
    assert len(result.shape) == 6
    assert result.shape[0] == T
    assert result.shape[1] == 3


def test_fvc_grad_linear_apply_linear_field_exact():
    """Test FVCGradLinear apply with linear field - exact values."""
    # Linear field: psi = x (in x direction)
    # Gradient should be (1.0, 0.0, 0.0)

    T, C, N, H = 1, 1, 4, 2
    device = torch.device("cpu")
    dtype = torch.float32
    cell_field = CellField(T, C, N, H, dtype, device)

    # Create linear field: psi[i] = i
    for i in range(N + 2 * H):
        cell_field.raw[0, 0, :, :, i] = float(i)

    dx = torch.tensor([1.0, 1.0, 1.0])
    result = FVCGradLinear.apply(cell_field, dx)

    # Expected: grad_x = 1.0, grad_y = 0.0, grad_z = 0.0
    expected_grad_x = torch.ones(N, N, N)
    expected_grad_y = torch.zeros(N, N, N)
    expected_grad_z = torch.zeros(N, N, N)

    torch.testing.assert_close(
        result[0, 0, 0],
        expected_grad_x,
    )  # x-component
    torch.testing.assert_close(
        result[0, 1, 0],
        expected_grad_y,
    )  # y-component
    torch.testing.assert_close(
        result[0, 2, 0],
        expected_grad_z,
    )  # z-component


def test_fvc_grad_linear_apply_3d_linear_field():
    """Test FVCGradLinear apply with 3D linear field."""
    # Linear field: psi = x + 2y + 3z
    # Gradient should be (1.0, 2.0, 3.0)

    T, C, N, H = 1, 1, 3, 2
    device = torch.device("cpu")
    dtype = torch.float32
    cell_field = CellField(T, C, N, H, dtype, device)

    # Create 3D linear field: psi[i,j,k] = i + 2*j + 3*k
    for k in range(N + 2 * H):
        for j in range(N + 2 * H):
            for i in range(N + 2 * H):
                cell_field.raw[0, 0, k, j, i] = float(i + 2 * j + 3 * k)

    dx = torch.tensor([1.0, 1.0, 1.0])
    result = FVCGradLinear.apply(cell_field, dx)

    # Expected: grad = (1.0, 2.0, 3.0)
    expected_grad_x = torch.ones(N, N, N)
    expected_grad_y = torch.full((N, N, N), 2.0)
    expected_grad_z = torch.full((N, N, N), 3.0)

    torch.testing.assert_close(
        result[0, 0, 0],
        expected_grad_x,
    )
    torch.testing.assert_close(
        result[0, 1, 0],
        expected_grad_y,
    )
    torch.testing.assert_close(
        result[0, 2, 0],
        expected_grad_z,
    )
