from __future__ import annotations

import torch

from gridfoam.DNA.field import CellField
from gridfoam.DNA.scheme.fvc.div._linear import FVCLinearDiv


class TestFVCLinearDiv:
    """Test suite for FVCLinearDiv class."""

    def test_apply_constant_field(self) -> None:
        """Test apply with constant field (divergence should be zero)."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        div_op = FVCLinearDiv()
        psi_c = CellField(T, C, N, H, dtype, device)
        psi_c.raw[0] = (
            torch.ones(
                (C, N + 2 * H, N + 2 * H, N + 2 * H), dtype=dtype, device=device
            )
            * 5.0
        )

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        result = div_op.apply(psi_c, dx)

        assert result.shape == (T, C, N, N, N)
        # Divergence of constant field should be zero
        expected = torch.zeros((T, C, N, N, N), dtype=dtype, device=device)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_apply_linear_field_x(self) -> None:
        """Test apply with linear field varying in X direction."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        div_op = FVCLinearDiv()
        psi_c = CellField(T, C, N, H, dtype, device)
        # Create linear field: psi = x (varies along X axis)
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, 1, 1, -1
        )
        psi_c.raw = value.expand(T, C, N + 2 * H, N + 2 * H, -1)

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        result = div_op.apply(psi_c, dx)

        assert result.shape == (T, C, N, N, N)
        # For linear field psi = x, divergence should be 1.0
        # (since d/dx(x) = 1)
        expected = torch.ones((T, C, N, N, N), dtype=dtype, device=device)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_apply_linear_field_y(self) -> None:
        """Test apply with linear field varying in Y direction."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        div_op = FVCLinearDiv()
        psi_c = CellField(T, C, N, H, dtype, device)
        # Create linear field: psi = y (varies along Y axis)
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, 1, -1, 1
        )
        psi_c.raw = value.expand(T, C, N + 2 * H, -1, N + 2 * H)

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        result = div_op.apply(psi_c, dx)

        assert result.shape == (T, C, N, N, N)
        # For linear field psi = y, divergence should be 1.0
        # (since d/dy(y) = 1)
        expected = torch.ones((T, C, N, N, N), dtype=dtype, device=device)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_apply_linear_field_z(self) -> None:
        """Test apply with linear field varying in Z direction."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        div_op = FVCLinearDiv()
        psi_c = CellField(T, C, N, H, dtype, device)
        # Create linear field: psi = z (varies along Z axis)
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, -1, 1, 1
        )
        psi_c.raw = value.expand(T, C, -1, N + 2 * H, N + 2 * H)

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        result = div_op.apply(psi_c, dx)

        assert result.shape == (T, C, N, N, N)
        # For linear field psi = z, divergence should be 1.0
        # (since d/dz(z) = 1)
        expected = torch.ones((T, C, N, N, N), dtype=dtype, device=device)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_apply_linear_field_all_directions(self) -> None:
        """Test apply with linear field varying in all directions."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        div_op = FVCLinearDiv()
        psi_c = CellField(T, C, N, H, dtype, device)
        # Create linear field: psi = x + y + z
        x_val = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, 1, 1, -1
        )
        y_val = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, 1, -1, 1
        )
        z_val = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, -1, 1, 1
        )
        psi_c.raw = (
            x_val.expand(T, C, N + 2 * H, N + 2 * H, -1)
            + y_val.expand(T, C, N + 2 * H, -1, N + 2 * H)
            + z_val.expand(T, C, -1, N + 2 * H, N + 2 * H)
        )

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        result = div_op.apply(psi_c, dx)

        assert result.shape == (T, C, N, N, N)
        # For linear field psi = x + y + z, divergence should be 3.0
        # (since d/dx(x) + d/dy(y) + d/dz(z) = 1 + 1 + 1 = 3)
        expected = torch.ones((T, C, N, N, N), dtype=dtype, device=device) * 3.0
        assert torch.allclose(result, expected, atol=1e-5)

    def test_apply_different_dx(self) -> None:
        """Test apply with different dx values."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        div_op = FVCLinearDiv()
        psi_c = CellField(T, C, N, H, dtype, device)
        # Create linear field: psi = x
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, 1, 1, -1
        )
        psi_c.raw = value.expand(T, C, N + 2 * H, N + 2 * H, -1)

        dx = torch.tensor([2.0, 1.0, 1.0], dtype=dtype, device=device)

        result = div_op.apply(psi_c, dx)

        assert result.shape == (T, C, N, N, N)
        # For linear field psi = x with dx[0] = 2.0,
        # divergence should still be 1.0 (normalized by volume)
        # Sf = [1*1, 2*1, 2*1] = [1, 2, 2]
        # V = 2*1*1 = 2
        # The divergence calculation accounts for the grid spacing
        expected = 0.5 * torch.ones((T, C, N, N, N), dtype=dtype, device=device)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_apply_multiple_components(self) -> None:
        """Test apply with multiple components."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        div_op = FVCLinearDiv()
        psi_c = CellField(T, C, N, H, dtype, device)
        # Set different linear fields for each component
        for c in range(C):
            value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
                1, 1, 1, 1, -1
            ) * (c + 1)
            psi_c.raw[:, c, :, :, :] = value.expand(
                T, 1, N + 2 * H, N + 2 * H, -1
            )

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        result = div_op.apply(psi_c, dx)

        assert result.shape == (T, C, N, N, N)
        # Each component should have divergence proportional to its multiplier
        for c in range(C):
            expected = torch.ones((N, N, N), dtype=dtype, device=device) * (
                c + 1
            )
            assert torch.allclose(result[0, c, :, :, :], expected, atol=1e-5)

    def test_apply_multiple_time_levels(self) -> None:
        """Test apply with multiple time levels."""
        T, C, N, H = 2, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        div_op = FVCLinearDiv()
        psi_c = CellField(T, C, N, H, dtype, device)
        # Set different linear fields for each time level
        for t in range(T):
            value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
                1, 1, 1, 1, -1
            ) * (t + 1)
            psi_c.raw[t, :, :, :, :] = value.expand(
                1, 1, N + 2 * H, N + 2 * H, -1
            )

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        result = div_op.apply(psi_c, dx)

        assert result.shape == (T, C, N, N, N)
        # Each time level should have divergence proportional to its multiplier
        for t in range(T):
            expected = torch.ones((N, N, N), dtype=dtype, device=device) * (
                t + 1
            )
            assert torch.allclose(result[t, 0, :, :, :], expected, atol=1e-5)
