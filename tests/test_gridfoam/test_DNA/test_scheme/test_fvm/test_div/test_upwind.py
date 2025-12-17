from __future__ import annotations

import torch

from gridfoam.DNA.field import CellField, FaceField, FVMatrix
from gridfoam.DNA.scheme.fvm.div._upwind import FVMUpwindDiv


class TestFVMUpwindDiv:
    """Test suite for FVMUpwindDiv class."""

    def test_apply_basic(self) -> None:
        """Test basic apply functionality."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        div_op = FVMUpwindDiv()
        phi_f = FaceField(T, C, N, dtype, device)
        psi_c = CellField(T, C, N, H, dtype, device)

        # Set zero flux
        phi_f.x.fill_(0.0)
        phi_f.y.fill_(0.0)
        phi_f.z.fill_(0.0)

        dt = 0.1
        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        fvmatrix = div_op.apply(phi_f, psi_c, dt, dx)

        assert isinstance(fvmatrix, FVMatrix)
        assert fvmatrix.C == C
        assert fvmatrix.N == N
        assert fvmatrix.H == H

        # All coefficients should be zero for zero flux
        expected_zero = torch.zeros((C, N, N, N), dtype=dtype, device=device)
        assert torch.allclose(fvmatrix.a_P.interior[0], expected_zero)
        assert torch.allclose(fvmatrix.a_E.interior[0], expected_zero)
        assert torch.allclose(fvmatrix.a_W.interior[0], expected_zero)
        assert torch.allclose(fvmatrix.a_N.interior[0], expected_zero)
        assert torch.allclose(fvmatrix.a_S.interior[0], expected_zero)
        assert torch.allclose(fvmatrix.a_T.interior[0], expected_zero)
        assert torch.allclose(fvmatrix.a_B.interior[0], expected_zero)

    def test_apply_positive_flux_x(self) -> None:
        """Test apply with positive flux in X direction (outflow)."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        div_op = FVMUpwindDiv()
        phi_f = FaceField(T, C, N, dtype, device)
        psi_c = CellField(T, C, N, H, dtype, device)

        # Set positive flux in X direction (outflow)
        phi_f.x.fill_(1.0)
        phi_f.y.fill_(0.0)
        phi_f.z.fill_(0.0)

        dt = 0.1
        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)
        V = dx[0] * dx[1] * dx[2]

        fvmatrix = div_op.apply(phi_f, psi_c, dt, dx)

        # For positive flux phi_e = 1.0, phi_w = 1.0
        # a_P should have: max(phi_e, 0) - min(phi_w, 0) = 1.0 - 0 = 1.0
        # But phi_w is from interior faces, so it's phi_f.x[:, :, :, :, :-1]
        # phi_e is phi_f.x[:, :, :, :, 1:]
        # For constant flux of 1.0:
        # phi_e = 1.0, phi_w = 1.0
        # a_P = (max(1.0, 0) - min(1.0, 0)) / V = (1.0 - 0) / V = 1.0 / V
        expected_a_P = torch.ones((C, N, N, N), dtype=dtype, device=device) / V
        assert torch.allclose(fvmatrix.a_P.interior[0], expected_a_P, atol=1e-5)

    def test_apply_negative_flux_x(self) -> None:
        """Test apply with negative flux in X direction (inflow)."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        div_op = FVMUpwindDiv()
        phi_f = FaceField(T, C, N, dtype, device)
        psi_c = CellField(T, C, N, H, dtype, device)

        # Set negative flux in X direction (inflow)
        phi_f.x.fill_(-1.0)
        phi_f.y.fill_(0.0)
        phi_f.z.fill_(0.0)

        dt = 0.1
        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)
        V = dx[0] * dx[1] * dx[2]

        fvmatrix = div_op.apply(phi_f, psi_c, dt, dx)

        # For negative flux phi_e = -1.0, phi_w = -1.0
        # a_E should have: min(phi_e, 0) / V = min(-1.0, 0) / V = -1.0 / V
        # a_W should have: -max(phi_w, 0) / V = -max(-1.0, 0) / V = 0
        # a_P should have: max(phi_e, 0) - min(phi_w, 0) = 0 - (-1.0) = 1.0
        expected_a_E = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * (-1.0) / V
        )
        expected_a_P = torch.ones((C, N, N, N), dtype=dtype, device=device) / V
        assert torch.allclose(fvmatrix.a_E.interior[0], expected_a_E, atol=1e-5)
        assert torch.allclose(fvmatrix.a_P.interior[0], expected_a_P, atol=1e-5)

    def test_apply_positive_flux_y(self) -> None:
        """Test apply with positive flux in Y direction."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        div_op = FVMUpwindDiv()
        phi_f = FaceField(T, C, N, dtype, device)
        psi_c = CellField(T, C, N, H, dtype, device)

        # Set positive flux in Y direction
        phi_f.x.fill_(0.0)
        phi_f.y.fill_(2.0)
        phi_f.z.fill_(0.0)

        dt = 0.1
        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)
        V = dx[0] * dx[1] * dx[2]

        fvmatrix = div_op.apply(phi_f, psi_c, dt, dx)

        # For positive flux in Y: phi_n = 2.0, phi_s = 2.0
        # a_P should have: max(phi_n, 0) - min(phi_s, 0) = 2.0 - 0 = 2.0
        expected_a_P = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 2.0 / V
        )
        assert torch.allclose(fvmatrix.a_P.interior[0], expected_a_P, atol=1e-5)

    def test_apply_positive_flux_z(self) -> None:
        """Test apply with positive flux in Z direction."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        div_op = FVMUpwindDiv()
        phi_f = FaceField(T, C, N, dtype, device)
        psi_c = CellField(T, C, N, H, dtype, device)

        # Set positive flux in Z direction
        phi_f.x.fill_(0.0)
        phi_f.y.fill_(0.0)
        phi_f.z.fill_(3.0)

        dt = 0.1
        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)
        V = dx[0] * dx[1] * dx[2]

        fvmatrix = div_op.apply(phi_f, psi_c, dt, dx)

        # For positive flux in Z: phi_t = 3.0, phi_b = 3.0
        # a_P should have: max(phi_t, 0) - min(phi_b, 0) = 3.0 - 0 = 3.0
        expected_a_P = (
            torch.ones((C, N, N, N), dtype=dtype, device=device) * 3.0 / V
        )
        assert torch.allclose(fvmatrix.a_P.interior[0], expected_a_P, atol=1e-5)

    def test_apply_mixed_flux(self) -> None:
        """Test apply with mixed positive and negative flux."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        div_op = FVMUpwindDiv()
        phi_f = FaceField(T, C, N, dtype, device)
        psi_c = CellField(T, C, N, H, dtype, device)

        psi_c.raw.fill_(1.0)

        # Set positive flux in X, negative in Y
        phi_f.x.fill_(1.0)
        phi_f.y.fill_(-1.0)
        phi_f.z.fill_(0.0)

        dt = 0.1
        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        fvmatrix = div_op.apply(phi_f, psi_c, dt, dx)

        # X direction: positive flux -> contributes to a_P
        # Y direction: negative flux -> contributes to a_N and a_P
        # a_P should have contributions from both directions
        assert fvmatrix.a_P.interior[0].sum() > 0
        assert fvmatrix.a_N.interior[0].sum() < 0  # Negative flux

        applied = fvmatrix.apply(psi_c)
        assert torch.allclose(
            applied, torch.zeros((C, N, N, N), dtype=dtype, device=device)
        )

    def test_apply_different_dx(self) -> None:
        """Test apply with different dx values."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        div_op = FVMUpwindDiv()
        phi_f = FaceField(T, C, N, dtype, device)
        psi_c = CellField(T, C, N, H, dtype, device)

        # Set positive flux
        phi_f.x.fill_(1.0)
        phi_f.y.fill_(0.0)
        phi_f.z.fill_(0.0)

        dt = 0.1
        dx1 = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)
        dx2 = torch.tensor([2.0, 1.0, 1.0], dtype=dtype, device=device)

        fvmatrix1 = div_op.apply(phi_f, psi_c, dt, dx1)
        fvmatrix2 = div_op.apply(phi_f, psi_c, dt, dx2)

        # Larger volume should give smaller coefficients
        assert torch.all(fvmatrix1.a_P.interior[0] > fvmatrix2.a_P.interior[0])

    def test_apply_multiple_components(self) -> None:
        """Test apply with multiple components."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        div_op = FVMUpwindDiv()
        phi_f = FaceField(T, C, N, dtype, device)
        psi_c = CellField(T, C, N, H, dtype, device)

        # Set different flux for each component
        phi_f.x[0, 0, :, :, :] = 1.0
        phi_f.x[0, 1, :, :, :] = 2.0
        phi_f.y.fill_(0.0)
        phi_f.z.fill_(0.0)

        dt = 0.1
        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)
        V = dx[0] * dx[1] * dx[2]

        fvmatrix = div_op.apply(phi_f, psi_c, dt, dx)

        assert fvmatrix.C == C
        # Each component should have different coefficients
        assert torch.allclose(
            fvmatrix.a_P.interior[0][0, :, :, :],
            torch.ones((N, N, N), dtype=dtype, device=device) / V,
            atol=1e-5,
        )
        assert torch.allclose(
            fvmatrix.a_P.interior[0][1, :, :, :],
            torch.ones((N, N, N), dtype=dtype, device=device) * 2.0 / V,
            atol=1e-5,
        )
