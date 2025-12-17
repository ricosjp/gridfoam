from __future__ import annotations

import torch

from gridfoam.DNA.field import CellField, FVMatrix


class TestFVMatrix:
    """Test suite for FVMatrix class."""

    def test_init(self) -> None:
        """Test FVMatrix initialization."""
        C, N, H = 3, 4, 1
        dtype = torch.float32
        device = torch.device("cpu")

        fvmatrix = FVMatrix(C, N, H, dtype, device)

        assert fvmatrix.C == C
        assert fvmatrix.N == N
        assert fvmatrix.H == H
        assert isinstance(fvmatrix.a_P, CellField)
        assert isinstance(fvmatrix.a_E, CellField)
        assert isinstance(fvmatrix.a_W, CellField)
        assert isinstance(fvmatrix.a_N, CellField)
        assert isinstance(fvmatrix.a_S, CellField)
        assert isinstance(fvmatrix.a_T, CellField)
        assert isinstance(fvmatrix.a_B, CellField)
        assert isinstance(fvmatrix.source, CellField)

    def test_properties(self) -> None:
        """Test FVMatrix properties."""
        C, N, H = 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        fvmatrix = FVMatrix(C, N, H, dtype, device)

        assert fvmatrix.C == C
        assert fvmatrix.N == N
        assert fvmatrix.H == H
        assert fvmatrix.a_P.C == C
        assert fvmatrix.a_P.N == N
        assert fvmatrix.a_P.H == H

    def test_a_P_setter(self) -> None:
        """Test a_P setter."""
        C, N, H = 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        fvmatrix = FVMatrix(C, N, H, dtype, device)
        value = torch.ones((C, N, N, N), dtype=dtype, device=device) * 5.0

        fvmatrix.a_P = value

        assert torch.allclose(fvmatrix.a_P.interior[0], value)

    def test_a_E_setter(self) -> None:
        """Test a_E setter."""
        C, N, H = 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        fvmatrix = FVMatrix(C, N, H, dtype, device)
        value = torch.ones((C, N, N, N), dtype=dtype, device=device) * 6.0

        fvmatrix.a_E = value

        assert torch.allclose(fvmatrix.a_E.interior[0], value)

    def test_a_W_setter(self) -> None:
        """Test a_W setter."""
        C, N, H = 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        fvmatrix = FVMatrix(C, N, H, dtype, device)
        value = torch.ones((C, N, N, N), dtype=dtype, device=device) * 7.0

        fvmatrix.a_W = value

        assert torch.allclose(fvmatrix.a_W.interior[0], value)

    def test_a_N_setter(self) -> None:
        """Test a_N setter."""
        C, N, H = 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        fvmatrix = FVMatrix(C, N, H, dtype, device)
        value = torch.ones((C, N, N, N), dtype=dtype, device=device) * 8.0

        fvmatrix.a_N = value

        assert torch.allclose(fvmatrix.a_N.interior[0], value)

    def test_a_S_setter(self) -> None:
        """Test a_S setter."""
        C, N, H = 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        fvmatrix = FVMatrix(C, N, H, dtype, device)
        value = torch.ones((C, N, N, N), dtype=dtype, device=device) * 9.0

        fvmatrix.a_S = value

        assert torch.allclose(fvmatrix.a_S.interior[0], value)

    def test_a_T_setter(self) -> None:
        """Test a_T setter."""
        C, N, H = 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        fvmatrix = FVMatrix(C, N, H, dtype, device)
        value = torch.ones((C, N, N, N), dtype=dtype, device=device) * 10.0

        fvmatrix.a_T = value

        assert torch.allclose(fvmatrix.a_T.interior[0], value)

    def test_a_B_setter(self) -> None:
        """Test a_B setter."""
        C, N, H = 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        fvmatrix = FVMatrix(C, N, H, dtype, device)
        value = torch.ones((C, N, N, N), dtype=dtype, device=device) * 11.0

        fvmatrix.a_B = value

        assert torch.allclose(fvmatrix.a_B.interior[0], value)

    def test_source_setter(self) -> None:
        """Test source setter."""
        C, N, H = 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        fvmatrix = FVMatrix(C, N, H, dtype, device)
        value = torch.ones((C, N, N, N), dtype=dtype, device=device) * 12.0

        fvmatrix.source = value

        assert torch.allclose(fvmatrix.source.interior[0], value)

    def test_iadd(self) -> None:
        """Test addition operator."""
        C, N, H = 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        matrix1 = FVMatrix(C, N, H, dtype, device)
        matrix2 = FVMatrix(C, N, H, dtype, device)

        matrix1.a_P = torch.ones((C, N, N, N), dtype=dtype, device=device) * 1.0
        matrix2.a_P = torch.ones((C, N, N, N), dtype=dtype, device=device) * 2.0

        matrix1 += matrix2

        expected = torch.ones((C, N, N, N), dtype=dtype, device=device) * 3.0
        assert torch.allclose(matrix1.a_P.interior[0], expected)

    def test_isub(self) -> None:
        """Test subtraction operator."""
        C, N, H = 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        matrix1 = FVMatrix(C, N, H, dtype, device)
        matrix2 = FVMatrix(C, N, H, dtype, device)

        matrix1.a_P = torch.ones((C, N, N, N), dtype=dtype, device=device) * 5.0
        matrix2.a_P = torch.ones((C, N, N, N), dtype=dtype, device=device) * 2.0

        matrix1 -= matrix2

        expected = torch.ones((C, N, N, N), dtype=dtype, device=device) * 3.0
        assert torch.allclose(matrix1.a_P.interior[0], expected)

    def test_add_all_coefficients(self) -> None:
        """Test addition operator for all coefficients."""
        C, N, H = 1, 2, 1
        dtype = torch.float32
        device = torch.device("cpu")

        matrix1 = FVMatrix(C, N, H, dtype, device)
        matrix2 = FVMatrix(C, N, H, dtype, device)

        # Set all coefficients
        matrix1.a_P = torch.ones((C, N, N, N), dtype=dtype, device=device) * 1.0
        matrix1.a_E = torch.ones((C, N, N, N), dtype=dtype, device=device) * 2.0
        matrix1.a_W = torch.ones((C, N, N, N), dtype=dtype, device=device) * 3.0
        matrix1.a_N = torch.ones((C, N, N, N), dtype=dtype, device=device) * 4.0
        matrix1.a_S = torch.ones((C, N, N, N), dtype=dtype, device=device) * 5.0
        matrix1.a_T = torch.ones((C, N, N, N), dtype=dtype, device=device) * 6.0
        matrix1.a_B = torch.ones((C, N, N, N), dtype=dtype, device=device) * 7.0
        matrix1.source = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 8.0
        )

        matrix2.a_P = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 10.0
        )
        matrix2.a_E = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 20.0
        )
        matrix2.a_W = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 30.0
        )
        matrix2.a_N = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 40.0
        )
        matrix2.a_S = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 50.0
        )
        matrix2.a_T = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 60.0
        )
        matrix2.a_B = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 70.0
        )
        matrix2.source = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 80.0
        )

        matrix1 += matrix2

        assert torch.allclose(
            matrix1.a_P.interior[0],
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 11.0,
        )
        assert torch.allclose(
            matrix1.a_E.interior[0],
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 22.0,
        )
        assert torch.allclose(
            matrix1.a_W.interior[0],
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 33.0,
        )
        assert torch.allclose(
            matrix1.a_N.interior[0],
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 44.0,
        )
        assert torch.allclose(
            matrix1.a_S.interior[0],
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 55.0,
        )
        assert torch.allclose(
            matrix1.a_T.interior[0],
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 66.0,
        )
        assert torch.allclose(
            matrix1.a_B.interior[0],
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 77.0,
        )
        assert torch.allclose(
            matrix1.source.interior[0],
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 88.0,
        )

    def test_a_fx(self) -> None:
        """Test a_fx property (harmonic mean along X axis)."""
        C, N, H = 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        fvmatrix = FVMatrix(C, N, H, dtype, device)
        # Set a_P with values that vary along X axis
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, 1, 1, -1
        )
        fvmatrix.a_P.raw[:, :, H:-H, H:-H, :] = value.expand(1, C, N, N, -1)

        a_fx = fvmatrix.a_fx

        assert a_fx.shape == (C, N, N, N + 1)
        # Harmonic mean of consecutive values
        # For values [-1, 0, 1, 2, 3] at positions [-1, 0, 1, 2, 3]
        # Face values should be harmonic mean of (-1,0), (0,1), (1,2), (2,3)
        # harmonic_mean(-1, 0) = 2*-1*0/(-1+0) = 0
        # harmonic_mean(0, 1) = 2*0*1/(0+1) = 0
        # harmonic_mean(1, 2) = 2*1*2/(1+2) = 4/3 ≈ 1.333
        # harmonic_mean(2, 3) = 2*2*3/(2+3) = 12/5 = 2.4
        expected = torch.zeros((C, N, N, N + 1), dtype=dtype, device=device)
        expected[:, :, :, 0] = 0.0
        expected[:, :, :, 1] = 0.0
        expected[:, :, :, 2] = 4.0 / 3.0
        expected[:, :, :, 3] = 12.0 / 5.0
        assert torch.allclose(a_fx, expected, atol=1e-5)

    def test_a_fy(self) -> None:
        """Test a_fy property (harmonic mean along Y axis)."""
        C, N, H = 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        fvmatrix = FVMatrix(C, N, H, dtype, device)
        # Set a_P with values that vary along Y axis
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, 1, -1, 1
        )
        fvmatrix.a_P.raw[:, :, H:-H, :, H:-H] = value.expand(1, C, N, -1, N)

        a_fy = fvmatrix.a_fy

        assert a_fy.shape == (C, N, N + 1, N)
        # Similar harmonic mean calculation as a_fx but along Y axis
        expected = torch.zeros((C, N, N + 1, N), dtype=dtype, device=device)
        expected[:, :, 0, :] = 0.0
        expected[:, :, 1, :] = 0.0
        expected[:, :, 2, :] = 4.0 / 3.0
        expected[:, :, 3, :] = 12.0 / 5.0
        assert torch.allclose(a_fy, expected, atol=1e-5)

    def test_a_fz(self) -> None:
        """Test a_fz property (harmonic mean along Z axis)."""
        C, N, H = 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        fvmatrix = FVMatrix(C, N, H, dtype, device)
        # Set a_P with values that vary along Z axis
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, -1, 1, 1
        )
        fvmatrix.a_P.raw[:, :, :, H:-H, H:-H] = value.expand(1, C, -1, N, N)

        a_fz = fvmatrix.a_fz

        assert a_fz.shape == (C, N + 1, N, N)
        # Similar harmonic mean calculation as a_fx but along Z axis
        expected = torch.zeros((C, N + 1, N, N), dtype=dtype, device=device)
        expected[:, 0, :, :] = 0.0
        expected[:, 1, :, :] = 0.0
        expected[:, 2, :, :] = 4.0 / 3.0
        expected[:, 3, :, :] = 12.0 / 5.0
        assert torch.allclose(a_fz, expected, atol=1e-5)

    def test_apply(self) -> None:
        """Test apply method."""
        C, N, H = 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        fvmatrix = FVMatrix(C, N, H, dtype, device)
        xi = CellField(1, C, N, H, dtype, device)

        # Set coefficients
        fvmatrix.a_P = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 2.0
        )
        fvmatrix.a_E = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 1.0
        )
        fvmatrix.a_W = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 1.0
        )
        fvmatrix.a_N = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 1.0
        )
        fvmatrix.a_S = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 1.0
        )
        fvmatrix.a_T = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 1.0
        )
        fvmatrix.a_B = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 1.0
        )

        # Set xi with constant value
        xi.raw[0] = (
            torch.ones(
                (C, N + 2 * H, N + 2 * H, N + 2 * H), dtype=dtype, device=device
            )
            * 3.0
        )

        yi = fvmatrix.apply(xi)

        assert yi.shape == (C, N, N, N)
        # Expected: a_P * xi + neighbors contributions
        # For constant xi=3.0 and constant coefficients:
        # yi = 2.0 * 3.0 + 1.0 * 3.0 * 6 (neighbors) = 6.0 + 18.0 = 24.0
        expected = torch.ones((C, N, N, N), dtype=dtype, device=device) * 24.0
        assert torch.allclose(yi, expected)

    def test_harmonic_mean_edge_cases(self) -> None:
        """Test harmonic mean with edge cases (zero values)."""
        C, N, H = 1, 2, 1
        dtype = torch.float32
        device = torch.device("cpu")

        fvmatrix = FVMatrix(C, N, H, dtype, device)
        # Set a_P with some zero values
        fvmatrix.a_P.interior[0] = torch.zeros(
            (C, N, N, N), dtype=dtype, device=device
        )

        a_fx = fvmatrix.a_fx

        assert a_fx.shape == (C, N, N, N + 1)
        # Harmonic mean of (0.0, 0.0) = 0.0
        expected_zero = torch.zeros((C, N, N, N + 1))
        assert torch.allclose(a_fx, expected_zero)
