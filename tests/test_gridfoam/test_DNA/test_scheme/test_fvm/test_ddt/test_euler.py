from __future__ import annotations

import torch

from gridfoam.DNA.field import CellField, FVMatrix
from gridfoam.DNA.scheme.fvm.ddt._euler import FVMEulerDdt


class TestFVMEulerDdt:
    """Test suite for FVMEulerDdt class."""

    def test_apply_basic(self) -> None:
        """Test basic apply functionality."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        ddt_op = FVMEulerDdt()
        x_c = CellField(T, C, N, H, dtype, device)
        x_c.interior[0] = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 5.0
        )

        dt = 0.1
        fvmatrix = ddt_op.apply(x_c, dt)

        assert isinstance(fvmatrix, FVMatrix)
        assert fvmatrix.C == C
        assert fvmatrix.N == N
        assert fvmatrix.H == H

        # Check a_P = 1.0 / dt
        rdt = 1.0 / dt
        expected_a_P = torch.full((C, N, N, N), rdt, dtype=dtype, device=device)
        assert torch.allclose(fvmatrix.a_P.interior[0], expected_a_P)

        # Check source = rdt * x_c.interior[0]
        expected_source = rdt * x_c.interior[0]
        assert torch.allclose(fvmatrix.source.interior[0], expected_source)

        # Check other coefficients are zero
        assert torch.allclose(
            fvmatrix.a_E.interior[0],
            torch.zeros((C, N, N, N), dtype=dtype, device=device),
        )
        assert torch.allclose(
            fvmatrix.a_W.interior[0],
            torch.zeros((C, N, N, N), dtype=dtype, device=device),
        )
        assert torch.allclose(
            fvmatrix.a_N.interior[0],
            torch.zeros((C, N, N, N), dtype=dtype, device=device),
        )
        assert torch.allclose(
            fvmatrix.a_S.interior[0],
            torch.zeros((C, N, N, N), dtype=dtype, device=device),
        )
        assert torch.allclose(
            fvmatrix.a_T.interior[0],
            torch.zeros((C, N, N, N), dtype=dtype, device=device),
        )
        assert torch.allclose(
            fvmatrix.a_B.interior[0],
            torch.zeros((C, N, N, N), dtype=dtype, device=device),
        )

    def test_apply_different_dt(self) -> None:
        """Test apply with different dt values."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        ddt_op = FVMEulerDdt()
        x_c = CellField(T, C, N, H, dtype, device)
        x_c.interior[0] = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 2.0
        )

        # Test with dt = 0.01
        dt1 = 0.01
        fvmatrix1 = ddt_op.apply(x_c, dt1)
        rdt1 = 1.0 / dt1
        expected_a_P1 = torch.full(
            (C, N, N, N), rdt1, dtype=dtype, device=device
        )
        assert torch.allclose(fvmatrix1.a_P.interior[0], expected_a_P1)

        # Test with dt = 1.0
        dt2 = 1.0
        fvmatrix2 = ddt_op.apply(x_c, dt2)
        rdt2 = 1.0 / dt2
        expected_a_P2 = torch.full(
            (C, N, N, N), rdt2, dtype=dtype, device=device
        )
        assert torch.allclose(fvmatrix2.a_P.interior[0], expected_a_P2)

        # Smaller dt should give larger a_P
        assert torch.all(fvmatrix1.a_P.interior[0] > fvmatrix2.a_P.interior[0])

    def test_apply_different_x_c_values(self) -> None:
        """Test apply with different x_c values."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        ddt_op = FVMEulerDdt()
        x_c = CellField(T, C, N, H, dtype, device)
        x_c.interior[0] = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 10.0
        )

        dt = 0.1
        fvmatrix = ddt_op.apply(x_c, dt)

        rdt = 1.0 / dt
        # a_P should be independent of x_c value
        expected_a_P = torch.full((C, N, N, N), rdt, dtype=dtype, device=device)
        assert torch.allclose(fvmatrix.a_P.interior[0], expected_a_P)

        # source should be rdt * x_c.interior[0]
        expected_source = rdt * x_c.interior[0]
        assert torch.allclose(fvmatrix.source.interior[0], expected_source)

    def test_apply_varying_x_c(self) -> None:
        """Test apply with varying x_c values."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        ddt_op = FVMEulerDdt()
        x_c = CellField(T, C, N, H, dtype, device)
        # Create varying values
        value = torch.arange(0, N * N * N, dtype=dtype, device=device).reshape(
            C, N, N, N
        )
        x_c.interior[0] = value

        dt = 0.1
        fvmatrix = ddt_op.apply(x_c, dt)

        rdt = 1.0 / dt
        # a_P should be constant
        expected_a_P = torch.full((C, N, N, N), rdt, dtype=dtype, device=device)
        assert torch.allclose(fvmatrix.a_P.interior[0], expected_a_P)

        # source should be rdt * x_c.interior[0]
        expected_source = rdt * x_c.interior[0]
        assert torch.allclose(fvmatrix.source.interior[0], expected_source)

    def test_apply_multiple_components(self) -> None:
        """Test apply with multiple components."""
        T, C, N, H = 1, 3, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        ddt_op = FVMEulerDdt()
        x_c = CellField(T, C, N, H, dtype, device)
        # Set different values for each component
        for c in range(C):
            x_c.interior[0][c, :, :, :] = float(c + 1) * 2.0

        dt = 0.1
        fvmatrix = ddt_op.apply(x_c, dt)

        assert fvmatrix.C == C
        rdt = 1.0 / dt
        # a_P should be constant for all components
        expected_a_P = torch.full((C, N, N, N), rdt, dtype=dtype, device=device)
        assert torch.allclose(fvmatrix.a_P.interior[0], expected_a_P)

        # source should be rdt * x_c.interior[0] for each component
        expected_source = rdt * x_c.interior[0]
        assert torch.allclose(fvmatrix.source.interior[0], expected_source)

    def test_apply_multiple_time_levels(self) -> None:
        """Test apply with multiple time levels."""
        T, C, N, H = 2, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        ddt_op = FVMEulerDdt()
        x_c = CellField(T, C, N, H, dtype, device)
        # Set different values for each time level
        for t in range(T):
            x_c.interior[t] = torch.ones(
                (C, N, N, N), dtype=dtype, device=device
            ) * (t + 1)

        dt = 0.1
        fvmatrix = ddt_op.apply(x_c, dt)

        rdt = 1.0 / dt
        # a_P should be constant
        expected_a_P = torch.full((C, N, N, N), rdt, dtype=dtype, device=device)
        assert torch.allclose(fvmatrix.a_P.interior[0], expected_a_P)

        # source should be rdt * x_c.interior[0] (first time level)
        expected_source = rdt * x_c.interior[0]
        assert torch.allclose(fvmatrix.source.interior[0], expected_source)
