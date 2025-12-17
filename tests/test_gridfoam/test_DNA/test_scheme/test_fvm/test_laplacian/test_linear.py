from __future__ import annotations

import pytest
import torch

from gridfoam.DNA.field import CellField, FVMatrix
from gridfoam.DNA.scheme.fvm.laplacian._linear import FVMLinearLaplacian


class TestFVMLinearLaplacian:
    """Test suite for FVMLinearLaplacian class."""

    def test_apply_constant_gamma(self) -> None:
        """Test apply with constant gamma (coefficients should be zero)."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        laplacian_op = FVMLinearLaplacian()
        gamma_c = CellField(T, C, N, H, dtype, device)
        psi_c = CellField(T, C, N, H, dtype, device)

        # Constant gamma
        gamma_c.raw.fill_(2.0)
        psi_c.raw.fill_(5.0)

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        fvmatrix = laplacian_op.apply(gamma_c, psi_c, dx)

        assert isinstance(fvmatrix, FVMatrix)
        assert fvmatrix.C == C
        assert fvmatrix.N == N

        # For constant gamma, all coefficients should be zero
        # since gamma_E - gamma_P = 0, etc.
        expected_zero = torch.zeros(
            (C, N, N, N), dtype=dtype, device=device
        )
        assert torch.allclose(
            fvmatrix.a_E.interior[0], expected_zero, atol=1e-5
        )
        assert torch.allclose(
            fvmatrix.a_W.interior[0], expected_zero, atol=1e-5
        )
        assert torch.allclose(
            fvmatrix.a_N.interior[0], expected_zero, atol=1e-5
        )
        assert torch.allclose(
            fvmatrix.a_S.interior[0], expected_zero, atol=1e-5
        )
        assert torch.allclose(
            fvmatrix.a_T.interior[0], expected_zero, atol=1e-5
        )
        assert torch.allclose(
            fvmatrix.a_B.interior[0], expected_zero, atol=1e-5
        )
        assert torch.allclose(
            fvmatrix.a_P.interior[0], expected_zero, atol=1e-5
        )

    def test_apply_linear_gamma_x(self) -> None:
        """Test apply with linear gamma varying in X direction."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        laplacian_op = FVMLinearLaplacian()
        gamma_c = CellField(T, C, N, H, dtype, device)
        psi_c = CellField(T, C, N, H, dtype, device)

        # Linear gamma: gamma = x (varies along X axis)
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, 1, 1, -1
        )
        gamma_c.raw = value.expand(T, C, N + 2 * H, N + 2 * H, -1)
        psi_c.raw.fill_(5.0)

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        fvmatrix = laplacian_op.apply(gamma_c, psi_c, dx)

        # For linear gamma = x:
        # gamma_E - gamma_P = 1, gamma_W - gamma_P = -1
        # a_E = Sf[0] * 0.5 * 1 / dx[0] = 1.0 * 0.5 * 1 / 1.0 = 0.5
        # a_W = Sf[0] * 0.5 * (-1) / dx[0] = 1.0 * 0.5 * (-1) / 1.0 = -0.5
        # a_P = -(a_E + a_W) = -(0.5 - 0.5) = 0
        # But in Y and Z directions, gamma is constant,
        # so a_N = a_S = a_T = a_B = 0
        # So a_P = -(0.5 - 0.5 + 0 + 0 + 0 + 0) = 0
        expected_a_E = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 0.5
        )
        expected_a_W = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * (-0.5)
        )
        # Note: boundary effects may cause differences at edges
        a_E_slice = fvmatrix.a_E.interior[0][:, :, :, :-1]
        expected_a_E_slice = expected_a_E[:, :, :, :-1]
        assert torch.allclose(a_E_slice, expected_a_E_slice, atol=1e-5)
        a_W_slice = fvmatrix.a_W.interior[0][:, :, :, 1:]
        expected_a_W_slice = expected_a_W[:, :, :, 1:]
        assert torch.allclose(a_W_slice, expected_a_W_slice, atol=1e-5)

    def test_apply_linear_gamma_y(self) -> None:
        """Test apply with linear gamma varying in Y direction."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        laplacian_op = FVMLinearLaplacian()
        gamma_c = CellField(T, C, N, H, dtype, device)
        psi_c = CellField(T, C, N, H, dtype, device)

        # Linear gamma: gamma = y (varies along Y axis)
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, 1, -1, 1
        )
        gamma_c.raw = value.expand(T, C, N + 2 * H, -1, N + 2 * H)
        psi_c.raw.fill_(5.0)

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        fvmatrix = laplacian_op.apply(gamma_c, psi_c, dx)

        # For linear gamma = y:
        # gamma_N - gamma_P = 1, gamma_S - gamma_P = -1
        # a_N = Sf[1] * 0.5 * 1 / dx[1] = 1.0 * 0.5 * 1 / 1.0 = 0.5
        # a_S = Sf[1] * 0.5 * (-1) / dx[1] = 1.0 * 0.5 * (-1) / 1.0 = -0.5
        expected_a_N = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 0.5
        )
        expected_a_S = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * (-0.5)
        )
        # Note: boundary effects may cause differences at edges
        assert torch.allclose(
            fvmatrix.a_N.interior[0][:, :, :-1, :],
            expected_a_N[:, :, :-1, :],
            atol=1e-5,
        )
        assert torch.allclose(
            fvmatrix.a_S.interior[0][:, :, 1:, :],
            expected_a_S[:, :, 1:, :],
            atol=1e-5,
        )

    def test_apply_linear_gamma_z(self) -> None:
        """Test apply with linear gamma varying in Z direction."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        laplacian_op = FVMLinearLaplacian()
        gamma_c = CellField(T, C, N, H, dtype, device)
        psi_c = CellField(T, C, N, H, dtype, device)

        # Linear gamma: gamma = z (varies along Z axis)
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, -1, 1, 1
        )
        gamma_c.raw = value.expand(T, C, -1, N + 2 * H, N + 2 * H)
        psi_c.raw.fill_(5.0)

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        fvmatrix = laplacian_op.apply(gamma_c, psi_c, dx)

        # For linear gamma = z:
        # gamma_T - gamma_P = 1, gamma_B - gamma_P = -1
        # a_T = Sf[2] * 0.5 * 1 / dx[2] = 1.0 * 0.5 * 1 / 1.0 = 0.5
        # a_B = Sf[2] * 0.5 * (-1) / dx[2] = 1.0 * 0.5 * (-1) / 1.0 = -0.5
        expected_a_T = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 0.5
        )
        expected_a_B = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * (-0.5)
        )
        # Note: boundary effects may cause differences at edges
        assert torch.allclose(
            fvmatrix.a_T.interior[0][:, :-1, :, :],
            expected_a_T[:, :-1, :, :],
            atol=1e-5,
        )
        assert torch.allclose(
            fvmatrix.a_B.interior[0][:, 1:, :, :],
            expected_a_B[:, 1:, :, :],
            atol=1e-5,
        )

    def test_apply_different_dx(self) -> None:
        """Test apply with different dx values."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        laplacian_op = FVMLinearLaplacian()
        gamma_c = CellField(T, C, N, H, dtype, device)
        psi_c = CellField(T, C, N, H, dtype, device)

        # Linear gamma: gamma = x
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, 1, 1, -1
        )
        gamma_c.raw = value.expand(T, C, N + 2 * H, N + 2 * H, -1)
        psi_c.raw.fill_(5.0)

        dx1 = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)
        dx2 = torch.tensor([2.0, 1.0, 1.0], dtype=dtype, device=device)

        fvmatrix1 = laplacian_op.apply(gamma_c, psi_c, dx1)
        fvmatrix2 = laplacian_op.apply(gamma_c, psi_c, dx2)

        # For dx[0] = 2.0, a_E and a_W should be half of dx[0] = 1.0 case
        # a_E = Sf[0] * 0.5 * 1 / dx[0]
        # For dx1: a_E = 1.0 * 0.5 * 1 / 1.0 = 0.5
        # For dx2: a_E = 1.0 * 0.5 * 1 / 2.0 = 0.25
        assert torch.all(
            fvmatrix1.a_E.interior[0][:, :, :, :-1]
            > fvmatrix2.a_E.interior[0][:, :, :, :-1]
        )

    def test_apply_scalar_field_constraint(self) -> None:
        """Test that apply only works with scalar fields (C=1)."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        laplacian_op = FVMLinearLaplacian()
        gamma_c = CellField(T, C, N, H, dtype, device)
        psi_c = CellField(T, C, N, H, dtype, device)

        gamma_c.raw.fill_(1.0)
        psi_c.raw.fill_(1.0)

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        # Should raise AssertionError for C != 1
        with pytest.raises(AssertionError):
            laplacian_op.apply(gamma_c, psi_c, dx)

    def test_apply_a_P_sum(self) -> None:
        """Test that a_P = -(a_E + a_W + a_N + a_S + a_T + a_B)."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        laplacian_op = FVMLinearLaplacian()
        gamma_c = CellField(T, C, N, H, dtype, device)
        psi_c = CellField(T, C, N, H, dtype, device)

        # Varying gamma
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, 1, 1, -1
        )
        gamma_c.raw = value.expand(T, C, N + 2 * H, N + 2 * H, -1)
        psi_c.raw.fill_(5.0)

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        fvmatrix = laplacian_op.apply(gamma_c, psi_c, dx)

        # Check that a_P = -(a_E + a_W + a_N + a_S + a_T + a_B)
        sum_neighbors = (
            fvmatrix.a_E.interior[0]
            + fvmatrix.a_W.interior[0]
            + fvmatrix.a_N.interior[0]
            + fvmatrix.a_S.interior[0]
            + fvmatrix.a_T.interior[0]
            + fvmatrix.a_B.interior[0]
        )
        expected_a_P = -sum_neighbors
        assert torch.allclose(fvmatrix.a_P.interior[0], expected_a_P, atol=1e-5)

