from __future__ import annotations

import pytest
import torch

from gridfoam.DNA.field import CellField
from gridfoam.DNA.scheme.fvc.laplacian._linear import FVCLinearLaplacian


class TestFVCLinearLaplacian:
    """Test suite for FVCLinearLaplacian class."""

    def test_apply_constant_fields(self) -> None:
        """Test apply with constant fields (laplacian should be zero)."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        laplacian_op = FVCLinearLaplacian()
        gamma_c = CellField(T, C, N, H, dtype, device)
        psi_c = CellField(T, C, N, H, dtype, device)

        gamma_c.raw[0] = (
            torch.ones(
                (C, N + 2 * H, N + 2 * H, N + 2 * H),
                dtype=dtype,
                device=device,
            )
            * 2.0
        )
        psi_c.raw[0] = (
            torch.ones(
                (C, N + 2 * H, N + 2 * H, N + 2 * H),
                dtype=dtype,
                device=device,
            )
            * 5.0
        )

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        result = laplacian_op.apply(gamma_c, psi_c, dx)

        assert result.shape == (T, C, N, N, N)
        # Laplacian of constant field should be zero
        expected = torch.zeros((T, C, N, N, N), dtype=dtype, device=device)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_apply_constant_gamma_linear_psi(self) -> None:
        """Test apply with constant gamma and linear psi."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        laplacian_op = FVCLinearLaplacian()
        gamma_c = CellField(T, C, N, H, dtype, device)
        psi_c = CellField(T, C, N, H, dtype, device)

        # Constant gamma
        gamma_c.raw[0] = (
            torch.ones(
                (C, N + 2 * H, N + 2 * H, N + 2 * H),
                dtype=dtype,
                device=device,
            )
            * 2.0
        )

        # Linear psi: psi = x
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, 1, 1, -1
        )
        psi_c.raw = value.expand(T, C, N + 2 * H, N + 2 * H, -1)

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        result = laplacian_op.apply(gamma_c, psi_c, dx)

        assert result.shape == (T, C, N, N, N)
        # For constant gamma and linear psi, laplacian should be zero
        # (since grad(psi) is constant, div(gamma * grad(psi)) = 0)
        expected = torch.zeros((T, C, N, N, N), dtype=dtype, device=device)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_apply_constant_gamma_quadratic_psi_x(self) -> None:
        """Test apply with constant gamma and quadratic psi in X."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        laplacian_op = FVCLinearLaplacian()
        gamma_c = CellField(T, C, N, H, dtype, device)
        psi_c = CellField(T, C, N, H, dtype, device)

        # Constant gamma
        gamma_c.raw[0] = (
            torch.ones(
                (C, N + 2 * H, N + 2 * H, N + 2 * H),
                dtype=dtype,
                device=device,
            )
            * 1.0
        )

        # Quadratic psi: psi = x^2 (approximately)
        # Create values that approximate x^2
        x_coords = torch.arange(-H, N + H, dtype=dtype, device=device)
        x_squared = x_coords**2
        value = x_squared.reshape(1, 1, 1, 1, -1)
        psi_c.raw = value.expand(T, C, N + 2 * H, N + 2 * H, -1)

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        result = laplacian_op.apply(gamma_c, psi_c, dx)

        assert result.shape == (T, C, N, N, N)
        # For psi = x^2, d^2(psi)/dx^2 = 2
        # With constant gamma = 1, laplacian should be approximately 2
        expected = 2.0 * torch.ones((N, N, N), dtype=dtype, device=device)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_apply_linear_gamma_constant_psi(self) -> None:
        """Test apply with linear gamma and constant psi."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        laplacian_op = FVCLinearLaplacian()
        gamma_c = CellField(T, C, N, H, dtype, device)
        psi_c = CellField(T, C, N, H, dtype, device)

        # Linear gamma: gamma = x
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, 1, 1, -1
        )
        gamma_c.raw = value.expand(T, C, N + 2 * H, N + 2 * H, -1)

        # Constant psi
        psi_c.raw[0] = (
            torch.ones(
                (C, N + 2 * H, N + 2 * H, N + 2 * H),
                dtype=dtype,
                device=device,
            )
            * 5.0
        )

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        result = laplacian_op.apply(gamma_c, psi_c, dx)

        assert result.shape == (T, C, N, N, N)
        # For constant psi, grad(psi) = 0, so laplacian should be zero
        expected = torch.zeros((T, C, N, N, N), dtype=dtype, device=device)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_apply_different_dx(self) -> None:
        """Test apply with different dx values."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        laplacian_op = FVCLinearLaplacian()
        gamma_c = CellField(T, C, N, H, dtype, device)
        psi_c = CellField(T, C, N, H, dtype, device)

        # Constant gamma
        gamma_c.raw[0] = (
            torch.ones(
                (C, N + 2 * H, N + 2 * H, N + 2 * H),
                dtype=dtype,
                device=device,
            )
            * 1.0
        )

        # Linear psi: psi = x
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, 1, 1, -1
        )
        psi_c.raw = value.expand(T, C, N + 2 * H, N + 2 * H, -1)

        dx = torch.tensor([2.0, 1.0, 1.0], dtype=dtype, device=device)

        result = laplacian_op.apply(gamma_c, psi_c, dx)

        assert result.shape == (T, C, N, N, N)
        # For linear psi, laplacian should be zero regardless of dx
        expected = torch.zeros((T, C, N, N, N), dtype=dtype, device=device)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_apply_multiple_time_levels(self) -> None:
        """Test apply with multiple time levels."""
        T, C, N, H = 2, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        laplacian_op = FVCLinearLaplacian()
        gamma_c = CellField(T, C, N, H, dtype, device)
        psi_c = CellField(T, C, N, H, dtype, device)

        # Constant gamma for all time levels
        gamma_c.raw = (
            torch.ones(
                (T, C, N + 2 * H, N + 2 * H, N + 2 * H),
                dtype=dtype,
                device=device,
            )
            * 1.0
        )

        # Different linear psi for each time level
        for t in range(T):
            value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
                1, 1, 1, 1, -1
            ) * (t + 1)
            psi_c.raw[t, :, :, :, :] = value.expand(
                1, 1, N + 2 * H, N + 2 * H, -1
            )

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        result = laplacian_op.apply(gamma_c, psi_c, dx)

        assert result.shape == (T, C, N, N, N)
        # For linear psi, laplacian should be zero for all time levels
        expected = torch.zeros((T, C, N, N, N), dtype=dtype, device=device)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_apply_scalar_field_constraint(self) -> None:
        """Test that apply only works with scalar fields (C=1)."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        laplacian_op = FVCLinearLaplacian()
        gamma_c = CellField(T, C, N, H, dtype, device)
        psi_c = CellField(T, C, N, H, dtype, device)

        gamma_c.raw[0] = torch.ones(
            (C, N + 2 * H, N + 2 * H, N + 2 * H),
            dtype=dtype,
            device=device,
        )
        psi_c.raw[0] = torch.ones(
            (C, N + 2 * H, N + 2 * H, N + 2 * H),
            dtype=dtype,
            device=device,
        )

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        # Should raise AssertionError for C != 1
        with pytest.raises(AssertionError):
            laplacian_op.apply(gamma_c, psi_c, dx)
