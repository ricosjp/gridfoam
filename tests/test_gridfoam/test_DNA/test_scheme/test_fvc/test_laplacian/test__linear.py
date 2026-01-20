"""Tests for FVC laplacian linear operator."""

import torch

from gridfoam.DNA.fielddata import CellField
from gridfoam.DNA.scheme.fvc.laplacian._linear import FVCLaplacianLinear


def test_fvc_laplacian_linear_apply():
    """Test FVCLaplacianLinear apply method."""
    # Create simple cell fields
    T, C, N, H = 1, 1, 4, 2
    device = torch.device("cpu")
    dtype = torch.float32
    gamma_field = CellField(T, C, N, H, dtype, device)
    gamma_field.raw[0] = torch.ones(T, C, N + 2 * H, N + 2 * H, N + 2 * H)
    psi_field = CellField(T, C, N, H, dtype, device)
    psi_field.raw[0] = torch.ones(T, C, N + 2 * H, N + 2 * H, N + 2 * H)

    dx = torch.tensor([1.0, 1.0, 1.0])
    result = FVCLaplacianLinear.apply(gamma_field, psi_field, dx)
    assert result.shape == (T, 1, N, N, N)
    # For uniform fields, laplacian should be zero
    torch.testing.assert_close(result[0], torch.zeros(1, N, N, N))


def test_fvc_laplacian_linear_apply_shape():
    """Test FVCLaplacianLinear apply preserves expected shape."""
    T, C, N = 2, 3, 5
    device = torch.device("cpu")
    dtype = torch.float32
    gamma_field = CellField(T, C, N, 1, dtype, device)  # H=1
    gamma_field.interior[0] = torch.randn(C, N, N, N)
    psi_field = CellField(T, C, N, 1, dtype, device)  # H=1
    psi_field.interior[0] = torch.randn(C, N, N, N)

    dx = torch.tensor([0.5, 0.5, 0.5])
    result = FVCLaplacianLinear.apply(gamma_field, psi_field, dx)
    assert result.shape == (T, 1, N, N, N)


def test_fvc_laplacian_linear_apply_quadratic_field():
    """Test FVCLaplacianLinear apply with quadratic field - exact values."""
    # Quadratic field: psi = x^2, gamma = 1.0
    # Laplacian of x^2 is 2.0 (in 1D)
    # In 3D with uniform gamma, laplacian = 2.0 * 3 = 6.0

    T, C, N, H = 1, 1, 4, 2
    device = torch.device("cpu")
    dtype = torch.float32
    gamma_field = CellField(T, C, N, H, dtype, device)
    gamma_field.raw[0] = torch.ones(T, C, N + 2 * H, N + 2 * H, N + 2 * H)

    psi_field = CellField(T, C, N, H, dtype, device)
    # Create quadratic field: psi[i] = i^2
    for i in range(N + 2 * H):
        psi_field.raw[0, 0, :, :, i] = float(i * i)

    dx = torch.tensor([1.0, 1.0, 1.0])
    result = FVCLaplacianLinear.apply(gamma_field, psi_field, dx)

    # Expected: laplacian of x^2 is 2.0 in x direction
    # For 3D quadratic field x^2 + y^2 + z^2, laplacian = 6.0
    # But we only have x^2 variation, so laplacian should be approximately 2.0
    # Allow tolerance due to face averaging and boundary effects
    expected = torch.full((1, N, N, N), 2.0)
    torch.testing.assert_close(
        result[0],
        expected,
    )


def test_fvc_laplacian_linear_apply_linear_field():
    """Test FVCLaplacianLinear apply with linear field."""
    # Linear field: psi = x, gamma = 1.0
    # Laplacian of x is 0.0

    T, C, N, H = 1, 1, 4, 2
    device = torch.device("cpu")
    dtype = torch.float32
    gamma_field = CellField(T, C, N, H, dtype, device)
    gamma_field.raw[0] = torch.ones(T, C, N + 2 * H, N + 2 * H, N + 2 * H)

    psi_field = CellField(T, C, N, H, dtype, device)
    # Create linear field: psi[i] = i
    for i in range(N + 2 * H):
        psi_field.raw[0, 0, :, :, i] = float(i)

    dx = torch.tensor([1.0, 1.0, 1.0])
    result = FVCLaplacianLinear.apply(gamma_field, psi_field, dx)

    # Expected: laplacian of linear field should be zero
    expected = torch.zeros(1, N, N, N)
    torch.testing.assert_close(
        result[0],
        expected,
    )
