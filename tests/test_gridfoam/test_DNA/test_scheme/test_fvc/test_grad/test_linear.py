from __future__ import annotations

import torch

from gridfoam.DNA.field import CellField
from gridfoam.DNA.scheme.fvc.grad._linear import FVCLinearGrad


class TestFVCLinearGrad:
    """Test suite for FVCLinearGrad class."""

    def test_apply_constant_field(self) -> None:
        """Test apply with constant field (gradient should be zero)."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        grad_op = FVCLinearGrad()
        psi_c = CellField(T, C, N, H, dtype, device)
        psi_c.raw[0] = (
            torch.ones(
                (C, N + 2 * H, N + 2 * H, N + 2 * H), dtype=dtype, device=device
            )
            * 5.0
        )

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        result = grad_op.apply(psi_c, dx)

        assert result.shape == (T, 3, C, N, N, N)
        # Gradient of constant field should be zero
        expected = torch.zeros((T, 3, C, N, N, N), dtype=dtype, device=device)
        assert torch.allclose(result, expected)

    def test_apply_linear_field_x(self) -> None:
        """Test apply with linear field varying in X direction."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        grad_op = FVCLinearGrad()
        psi_c = CellField(T, C, N, H, dtype, device)
        # Create linear field: psi = x (varies along X axis)
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, 1, 1, -1
        )
        psi_c.raw = value.expand(T, C, N + 2 * H, N + 2 * H, -1)

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        result = grad_op.apply(psi_c, dx)

        assert result.shape == (T, 3, C, N, N, N)
        # Gradient in X direction should be approximately 1.0
        # (since field increases by 1.0 per cell with dx=1.0)
        grad_x = result[0, 0, 0, :, :, :]
        # For linear field psi = x, gradient should be 1.0
        expected_grad_x = torch.ones((N, N, N), dtype=dtype, device=device)
        assert torch.allclose(grad_x, expected_grad_x, atol=1e-5)

        # Gradients in Y and Z directions should be zero
        grad_y = result[0, 1, 0, :, :, :]
        grad_z = result[0, 2, 0, :, :, :]
        expected_zero = torch.zeros((N, N, N), dtype=dtype, device=device)
        assert torch.allclose(grad_y, expected_zero, atol=1e-5)
        assert torch.allclose(grad_z, expected_zero, atol=1e-5)

    def test_apply_linear_field_y(self) -> None:
        """Test apply with linear field varying in Y direction."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        grad_op = FVCLinearGrad()
        psi_c = CellField(T, C, N, H, dtype, device)
        # Create linear field: psi = y (varies along Y axis)
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, 1, -1, 1
        )
        psi_c.raw = value.expand(T, C, N + 2 * H, -1, N + 2 * H)

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        result = grad_op.apply(psi_c, dx)

        assert result.shape == (T, 3, C, N, N, N)
        # Gradient in Y direction should be approximately 1.0
        grad_y = result[0, 1, 0, :, :, :]
        expected_grad_y = torch.ones((N, N, N), dtype=dtype, device=device)
        assert torch.allclose(grad_y, expected_grad_y, atol=1e-5)

        # Gradients in X and Z directions should be zero
        grad_x = result[0, 0, 0, :, :, :]
        grad_z = result[0, 2, 0, :, :, :]
        expected_zero = torch.zeros((N, N, N), dtype=dtype, device=device)
        assert torch.allclose(grad_x, expected_zero, atol=1e-5)
        assert torch.allclose(grad_z, expected_zero, atol=1e-5)

    def test_apply_linear_field_z(self) -> None:
        """Test apply with linear field varying in Z direction."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        grad_op = FVCLinearGrad()
        psi_c = CellField(T, C, N, H, dtype, device)
        # Create linear field: psi = z (varies along Z axis)
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, -1, 1, 1
        )
        psi_c.raw = value.expand(T, C, -1, N + 2 * H, N + 2 * H)

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        result = grad_op.apply(psi_c, dx)

        assert result.shape == (T, 3, C, N, N, N)
        # Gradient in Z direction should be approximately 1.0
        grad_z = result[0, 2, 0, :, :, :]
        expected_grad_z = torch.ones((N, N, N), dtype=dtype, device=device)
        assert torch.allclose(grad_z, expected_grad_z, atol=1e-5)

        # Gradients in X and Y directions should be zero
        grad_x = result[0, 0, 0, :, :, :]
        grad_y = result[0, 1, 0, :, :, :]
        expected_zero = torch.zeros((N, N, N), dtype=dtype, device=device)
        assert torch.allclose(grad_x, expected_zero, atol=1e-5)
        assert torch.allclose(grad_y, expected_zero, atol=1e-5)

    def test_apply_different_dx(self) -> None:
        """Test apply with different dx values."""
        T, C, N, H = 1, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        grad_op = FVCLinearGrad()
        psi_c = CellField(T, C, N, H, dtype, device)
        # Create linear field: psi = x
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, 1, 1, -1
        )
        psi_c.raw = value.expand(T, C, N + 2 * H, N + 2 * H, -1)

        dx = torch.tensor([2.0, 1.0, 1.0], dtype=dtype, device=device)

        result = grad_op.apply(psi_c, dx)

        assert result.shape == (T, 3, C, N, N, N)
        # Gradient in X direction should be 1.0 / 2.0 = 0.5
        # (since field increases by 1.0 per cell with dx=2.0)
        grad_x = result[0, 0, 0, :, :, :]
        expected_grad_x = (
            torch.ones((N, N, N), dtype=dtype, device=device) * 0.5
        )
        assert torch.allclose(grad_x, expected_grad_x, atol=1e-5)

    def test_apply_multiple_components(self) -> None:
        """Test apply with multiple components."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        grad_op = FVCLinearGrad()
        psi_c = CellField(T, C, N, H, dtype, device)
        # Set different values for each component
        for c in range(C):
            value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
                1, 1, 1, 1, -1
            ) * (c + 1)
            psi_c.raw[:, c, :, :, :] = value.expand(
                T, 1, N + 2 * H, N + 2 * H, -1
            )

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        result = grad_op.apply(psi_c, dx)

        assert result.shape == (T, 3, C, N, N, N)
        # Each component should have gradient proportional to its multiplier
        for c in range(C):
            grad_x = result[0, 0, c, :, :, :]
            expected = torch.ones((N, N, N), dtype=dtype, device=device) * (
                c + 1
            )
            assert torch.allclose(grad_x, expected, atol=1e-5)

    def test_apply_multiple_time_levels(self) -> None:
        """Test apply with multiple time levels."""
        T, C, N, H = 2, 1, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        grad_op = FVCLinearGrad()
        psi_c = CellField(T, C, N, H, dtype, device)
        # Set different values for each time level
        for t in range(T):
            value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
                1, 1, 1, 1, -1
            ) * (t + 1)
            psi_c.raw[t, :, :, :, :] = value.expand(
                1, C, N + 2 * H, N + 2 * H, -1
            )

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        result = grad_op.apply(psi_c, dx)

        assert result.shape == (T, 3, C, N, N, N)
        # Each time level should have gradient proportional to its multiplier
        for t in range(T):
            grad_x = result[t, 0, 0, :, :, :]
            expected = torch.ones((N, N, N), dtype=dtype, device=device) * (
                t + 1
            )
            assert torch.allclose(grad_x, expected, atol=1e-5)
