from __future__ import annotations

import numpy as np
import torch

from gridfoam.DNA.enum import Axis
from gridfoam.DNA.field import CellField


class TestCellField:
    """Test suite for CellField class."""

    def test_init(self) -> None:
        """Test CellField initialization."""
        T, C, N, H = 2, 3, 4, 1
        dtype = torch.float32
        device = torch.device("cpu")
        W = N + 2 * H

        cell_field = CellField(T, C, N, H, dtype, device)

        assert cell_field.T == T
        assert cell_field.C == C
        assert cell_field.N == N
        assert cell_field.H == H
        assert cell_field.raw.shape == (T, C, W, W, W)
        assert cell_field.raw.dtype == dtype
        assert cell_field.raw.device == device
        assert torch.allclose(cell_field.raw, torch.zeros_like(cell_field.raw))

    def test_properties(self) -> None:
        """Test CellField properties."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")
        W = N + 2 * H

        cell_field = CellField(T, C, N, H, dtype, device)

        # Test property getters
        assert cell_field.T == T
        assert cell_field.C == C
        assert cell_field.N == N
        assert cell_field.H == H

        # Test raw property setter
        new_raw = torch.ones((T, C, W, W, W), dtype=dtype, device=device)
        cell_field.raw = new_raw
        assert torch.allclose(cell_field.raw, new_raw)

        # Test interior property
        assert cell_field.interior.shape == (T, C, N, N, N)
        assert torch.allclose(cell_field.interior, torch.ones((T, C, N, N, N)))

    def test_interior_setter(self) -> None:
        """Test interior property setter."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        interior_value = (
            torch.ones((T, C, N, N, N), dtype=dtype, device=device) * 5.0
        )

        cell_field.interior = interior_value

        assert torch.allclose(cell_field.interior, interior_value)

    def test_zeros_like(self) -> None:
        """Test zeros_like class method."""
        T, C, N, H = 2, 3, 4, 1
        dtype = torch.float32
        device = torch.device("cpu")

        original = CellField(T, C, N, H, dtype, device)
        original.raw.fill_(1.0)

        zeros_field = CellField.zeros_like(original)

        assert zeros_field.T == original.T
        assert zeros_field.C == original.C
        assert zeros_field.N == original.N
        assert zeros_field.H == original.H
        assert zeros_field.raw.shape == original.raw.shape
        assert zeros_field.raw.dtype == original.raw.dtype
        assert zeros_field.raw.device == original.raw.device
        assert torch.allclose(
            zeros_field.raw, torch.zeros_like(zeros_field.raw)
        )

    def test_iadd(self) -> None:
        """Test in-place addition operator."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        field1 = CellField(T, C, N, H, dtype, device)
        field2 = CellField(T, C, N, H, dtype, device)

        field1.raw.fill_(1.0)
        field2.raw.fill_(2.0)

        field1 += field2

        assert torch.allclose(field1.raw, torch.full_like(field1.raw, 3.0))

    def test_isub(self) -> None:
        """Test in-place subtraction operator."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        field1 = CellField(T, C, N, H, dtype, device)
        field2 = CellField(T, C, N, H, dtype, device)

        field1.raw.fill_(5.0)
        field2.raw.fill_(2.0)

        field1 -= field2

        assert torch.allclose(field1.raw, torch.full_like(field1.raw, 3.0))

    def test_imul_with_cellfield(self) -> None:
        """Test in-place multiplication operator with CellField."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        field1 = CellField(T, C, N, H, dtype, device)
        field2 = CellField(T, C, N, H, dtype, device)

        field1.raw.fill_(2.0)
        field2.raw.fill_(3.0)

        field1 *= field2

        assert torch.allclose(field1.raw, torch.full_like(field1.raw, 6.0))

    def test_imul_with_float(self) -> None:
        """Test in-place multiplication operator with float."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        field1 = CellField(T, C, N, H, dtype, device)
        field1.raw.fill_(2.0)

        field1 *= 3.0

        assert torch.allclose(field1.raw, torch.full_like(field1.raw, 6.0))

    def test_add(self) -> None:
        """Test addition operator."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        field1 = CellField(T, C, N, H, dtype, device)
        field2 = CellField(T, C, N, H, dtype, device)

        field1.raw.fill_(1.0)
        field2.raw.fill_(2.0)

        result = field1 + field2

        assert torch.allclose(result.raw, torch.full_like(result.raw, 3.0))
        # Original fields should not be modified
        assert torch.allclose(field1.raw, torch.full_like(field1.raw, 1.0))
        assert torch.allclose(field2.raw, torch.full_like(field2.raw, 2.0))

    def test_sub(self) -> None:
        """Test subtraction operator."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        field1 = CellField(T, C, N, H, dtype, device)
        field2 = CellField(T, C, N, H, dtype, device)

        field1.raw.fill_(5.0)
        field2.raw.fill_(2.0)

        result = field1 - field2

        assert torch.allclose(result.raw, torch.full_like(result.raw, 3.0))

    def test_mul_with_cellfield(self) -> None:
        """Test multiplication operator with CellField."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        field1 = CellField(T, C, N, H, dtype, device)
        field2 = CellField(T, C, N, H, dtype, device)

        field1.raw.fill_(2.0)
        field2.raw.fill_(3.0)

        result = field1 * field2

        assert torch.allclose(result.raw, torch.full_like(result.raw, 6.0))

    def test_mul_with_float(self) -> None:
        """Test multiplication operator with float."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        field1 = CellField(T, C, N, H, dtype, device)
        field1.raw.fill_(2.0)

        result = field1 * 3.0

        assert torch.allclose(result.raw, torch.full_like(result.raw, 6.0))

    def test_rmul(self) -> None:
        """Test right multiplication operator."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        field1 = CellField(T, C, N, H, dtype, device)
        field1.raw.fill_(2.0)

        result = 3.0 * field1

        assert torch.allclose(result.raw, torch.full_like(result.raw, 6.0))

    def test_get_halo_along_x_forward(self) -> None:
        """Test getting halo along X axis (forward)."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        cell_field.raw[:, :, :, :, -H:] = 1.0  # Set forward halo

        halo = cell_field.get_halo_along(Axis.X, forward=True)

        assert halo.shape == (T, C, N, N, H)
        assert torch.allclose(halo, torch.ones((T, C, N, N, H)))

    def test_get_halo_along_x_backward(self) -> None:
        """Test getting halo along X axis (backward)."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        cell_field.raw[:, :, :, :, :H] = 2.0  # Set backward halo

        halo = cell_field.get_halo_along(Axis.X, forward=False)

        assert halo.shape == (T, C, N, N, H)
        assert torch.allclose(halo, torch.full((T, C, N, N, H), 2.0))

    def test_get_halo_along_y_forward(self) -> None:
        """Test getting halo along Y axis (forward)."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        cell_field.raw[:, :, :, -H:, :] = 3.0  # Set forward halo

        halo = cell_field.get_halo_along(Axis.Y, forward=True)

        assert halo.shape == (T, C, N, H, N)
        assert torch.allclose(halo, torch.full((T, C, N, H, N), 3.0))

    def test_get_halo_along_y_backward(self) -> None:
        """Test getting halo along Y axis (backward)."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        cell_field.raw[:, :, :, :H, :] = 4.0  # Set backward halo

        halo = cell_field.get_halo_along(Axis.Y, forward=False)

        assert halo.shape == (T, C, N, H, N)
        assert torch.allclose(halo, torch.full((T, C, N, H, N), 4.0))

    def test_get_halo_along_z_forward(self) -> None:
        """Test getting halo along Z axis (forward)."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        cell_field.raw[:, :, -H:, :, :] = 5.0  # Set forward halo

        halo = cell_field.get_halo_along(Axis.Z, forward=True)

        assert halo.shape == (T, C, H, N, N)
        assert torch.allclose(halo, torch.full((T, C, H, N, N), 5.0))

    def test_get_halo_along_z_backward(self) -> None:
        """Test getting halo along Z axis (backward)."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        cell_field.raw[:, :, :H, :, :] = 6.0  # Set backward halo

        halo = cell_field.get_halo_along(Axis.Z, forward=False)

        assert halo.shape == (T, C, H, N, N)
        assert torch.allclose(halo, torch.full((T, C, H, N, N), 6.0))

    def test_set_halo_along_x_forward(self) -> None:
        """Test setting halo along X axis (forward)."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        value = torch.ones((T, C, N, N, H), dtype=dtype, device=device) * 7.0

        cell_field.set_halo_along(Axis.X, forward=True, value=value)

        halo = cell_field.get_halo_along(Axis.X, forward=True)
        assert torch.allclose(halo, value)

    def test_set_halo_along_x_backward(self) -> None:
        """Test setting halo along X axis (backward)."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        value = torch.ones((T, C, N, N, H), dtype=dtype, device=device) * 8.0

        cell_field.set_halo_along(Axis.X, forward=False, value=value)

        halo = cell_field.get_halo_along(Axis.X, forward=False)
        assert torch.allclose(halo, value)

    def test_set_halo_along_y_forward(self) -> None:
        """Test setting halo along Y axis (forward)."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        value = torch.ones((T, C, N, H, N), dtype=dtype, device=device) * 9.0

        cell_field.set_halo_along(Axis.Y, forward=True, value=value)

        halo = cell_field.get_halo_along(Axis.Y, forward=True)
        assert torch.allclose(halo, value)

    def test_set_halo_along_y_backward(self) -> None:
        """Test setting halo along Y axis (backward)."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        value = torch.ones((T, C, N, H, N), dtype=dtype, device=device) * 10.0

        cell_field.set_halo_along(Axis.Y, forward=False, value=value)

        halo = cell_field.get_halo_along(Axis.Y, forward=False)
        assert torch.allclose(halo, value)

    def test_set_halo_along_z_forward(self) -> None:
        """Test setting halo along Z axis (forward)."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        value = torch.ones((T, C, H, N, N), dtype=dtype, device=device) * 11.0

        cell_field.set_halo_along(Axis.Z, forward=True, value=value)

        halo = cell_field.get_halo_along(Axis.Z, forward=True)
        assert torch.allclose(halo, value)

    def test_set_halo_along_z_backward(self) -> None:
        """Test setting halo along Z axis (backward)."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        value = torch.ones((T, C, H, N, N), dtype=dtype, device=device) * 12.0

        cell_field.set_halo_along(Axis.Z, forward=False, value=value)

        halo = cell_field.get_halo_along(Axis.Z, forward=False)
        assert torch.allclose(halo, value)

    def test_get_interior_halo_along_x_forward(self) -> None:
        """Test getting interior halo along X axis (forward)."""
        T, C, N, H = 1, 2, 4, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        cell_field.raw[:, :, :, :, -2 * H : -H] = 1.0  # Set interior halo

        interior_halo = cell_field.get_interior_halo_along(Axis.X, forward=True)

        assert interior_halo.shape == (T, C, N, N, H)
        assert torch.allclose(interior_halo, torch.ones((T, C, N, N, H)))

    def test_get_interior_halo_along_x_backward(self) -> None:
        """Test getting interior halo along X axis (backward)."""
        T, C, N, H = 1, 2, 4, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        cell_field.raw[:, :, :, :, H : 2 * H] = 2.0  # Set interior halo

        interior_halo = cell_field.get_interior_halo_along(
            Axis.X, forward=False
        )

        assert interior_halo.shape == (T, C, N, N, H)
        assert torch.allclose(interior_halo, torch.full((T, C, H, N, N), 2.0))

    def test_get_shifted_interior_along_x_forward(self) -> None:
        """Test getting shifted interior along X axis (forward)."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        cell_field.interior.fill_(1.0)
        # Set shifted region
        cell_field.raw[:, :, H:-H, H:-H, H + 1 : None] = 2.0

        shifted = cell_field.get_shifted_interior_along(Axis.X, shift=1)

        assert shifted.shape == (T, C, N, N, N)
        assert torch.allclose(shifted, torch.full((T, C, N, N, N), 2.0))

    def test_get_shifted_interior_along_x_backward(self) -> None:
        """Test getting shifted interior along X axis (backward)."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        cell_field.interior.fill_(1.0)
        # Set shifted region
        cell_field.raw[:, :, H:-H, H:-H, H - 1 : None] = 3.0

        shifted = cell_field.get_shifted_interior_along(Axis.X, shift=-1)

        assert shifted.shape == (T, C, N, N, N)
        assert torch.allclose(shifted, torch.full((T, C, N, N, N), 3.0))

    def test_get_shifted_interior_along_for_face_x(self) -> None:
        """Test getting shifted interior for face along X axis."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        cell_field.interior.fill_(1.0)
        # Set shifted region for face (includes one extra point)
        cell_field.raw[:, :, H:-H, H:-H, H:None] = 2.0

        shifted = cell_field.get_shifted_interior_along_for_face(
            Axis.X, shift=0
        )

        assert shifted.shape == (T, C, N, N, N + 1)
        expected = torch.full((T, C, N, N, N + 1), 2.0)
        assert torch.allclose(shifted, expected)

    def test_get_shifted_interior_along_for_face_y(self) -> None:
        """Test getting shifted interior for face along Y axis."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        cell_field.interior.fill_(1.0)
        # Set shifted region for face
        cell_field.raw[:, :, H:-H, H:None, H:-H] = 3.0

        shifted = cell_field.get_shifted_interior_along_for_face(
            Axis.Y, shift=0
        )

        assert shifted.shape == (T, C, N, N + 1, N)
        expected = torch.full((T, C, N, N + 1, N), 3.0)
        assert torch.allclose(shifted, expected)

    def test_get_shifted_interior_along_for_face_z(self) -> None:
        """Test getting shifted interior for face along Z axis."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        cell_field.interior.fill_(1.0)
        # Set shifted region for face
        cell_field.raw[:, :, H:None, H:-H, H:-H] = 4.0

        shifted = cell_field.get_shifted_interior_along_for_face(
            Axis.Z, shift=0
        )

        assert shifted.shape == (T, C, N + 1, N, N)
        expected = torch.full((T, C, N + 1, N, N), 4.0)
        assert torch.allclose(shifted, expected)

    def test_face_average_along_x(self) -> None:
        """Test face average along X axis."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        # Set forward and backward values
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, 1, 1, -1
        )
        cell_field.raw[:, :, H:-H, H:-H, :] = value.expand(T, C, N, N, -1)

        face_avg = cell_field.face_average_along(Axis.X)

        assert face_avg.shape == (T, C, N, N, N + 1)
        expected_value = torch.arange(
            -H + 0.5, N + 0.5, dtype=dtype, device=device
        ).reshape(1, 1, 1, 1, -1)
        expected = torch.zeros((T, C, N, N, N + 1), dtype=dtype, device=device)
        expected = expected_value.expand(T, C, N, N, -1)
        assert torch.allclose(face_avg, expected)

    def test_face_average_along_y(self) -> None:
        """Test face average along Y axis."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        # Set forward and backward values
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, 1, -1, 1
        )
        cell_field.raw[:, :, H:-H, :, H:-H] = value.expand(T, C, N, -1, N)

        face_avg = cell_field.face_average_along(Axis.Y)

        assert face_avg.shape == (T, C, N, N + 1, N)
        expected_value = torch.arange(
            -H + 0.5, N + 0.5, dtype=dtype, device=device
        ).reshape(1, 1, 1, -1, 1)
        expected = torch.zeros((T, C, N, N + 1, N), dtype=dtype, device=device)
        expected[:, :, :, :] = expected_value.expand(T, C, N, -1, N)
        assert torch.allclose(face_avg, expected)

    def test_face_average_along_z(self) -> None:
        """Test face average along Z axis."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        # Set forward and backward values
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, -1, 1, 1
        )
        cell_field.raw[:, :, :, H:-H, H:-H] = value.expand(T, C, -1, N, N)

        face_avg = cell_field.face_average_along(Axis.Z)

        assert face_avg.shape == (T, C, N + 1, N, N)
        expected_value = torch.arange(
            -H + 0.5, N + 0.5, dtype=dtype, device=device
        ).reshape(1, 1, -1, 1, 1)
        expected = torch.zeros((T, C, N + 1, N, N), dtype=dtype, device=device)
        expected[:, :, :, :] = expected_value.expand(T, C, -1, N, N)
        assert torch.allclose(face_avg, expected)

    def test_face_diff_along_x(self) -> None:
        """Test face difference along X axis."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        # Set forward and backward values
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, 1, 1, -1
        )
        cell_field.raw[:, :, H:-H, H:-H, :] = value.expand(T, C, N, N, -1)

        face_diff = cell_field.face_diff_along(Axis.X)

        assert face_diff.shape == (T, C, N, N, N + 1)
        expected = torch.full(
            (T, C, N, N, N + 1), 1.0, dtype=dtype, device=device
        )
        assert torch.allclose(face_diff, expected)

    def test_face_diff_along_y(self) -> None:
        """Test face difference along Y axis."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        # Set forward and backward values
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, 1, -1, 1
        )
        cell_field.raw[:, :, H:-H, :, H:-H] = value.expand(T, C, N, -1, N)

        face_diff = cell_field.face_diff_along(Axis.Y)

        assert face_diff.shape == (T, C, N, N + 1, N)
        expected = torch.full(
            (T, C, N, N + 1, N), 1.0, dtype=dtype, device=device
        )
        assert torch.allclose(face_diff, expected)

    def test_face_diff_along_z(self) -> None:
        """Test face difference along Z axis."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        # Set forward and backward values
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, -1, 1, 1
        )
        cell_field.raw[:, :, :, H:-H, H:-H] = value.expand(T, C, -1, N, N)

        face_diff = cell_field.face_diff_along(Axis.Z)

        assert face_diff.shape == (T, C, N + 1, N, N)
        expected = torch.full(
            (T, C, N + 1, N, N), 1.0, dtype=dtype, device=device
        )
        assert torch.allclose(face_diff, expected)

    def test_face_average(self) -> None:
        """Test face_average method."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        cell_field.interior.fill_(1.0)

        face_field = cell_field.face_average()

        assert face_field.T == T
        assert face_field.C == C
        assert face_field.N == N
        assert face_field.x.shape == (T, C, N, N, N + 1)
        assert face_field.y.shape == (T, C, N, N + 1, N)
        assert face_field.z.shape == (T, C, N + 1, N, N)

    def test_face_grad(self) -> None:
        """Test face_grad method."""
        T, C, N, H = 1, 2, 3, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        # Set forward and backward values
        value = torch.arange(-H, N + H, dtype=dtype, device=device).reshape(
            1, 1, 1, 1, -1
        )
        cell_field.raw[:, :, H:-H, H:-H, :] = value.expand(T, C, N, N, -1)

        dx = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        grad_field = cell_field.face_grad(dx)

        assert grad_field.T == T
        assert grad_field.C == C
        assert grad_field.N == N
        # Check that gradient in x direction is approximately 1.0
        # (since field increases by 1.0 per cell with dx=1.0)
        assert torch.allclose(
            grad_field.x[:, :, :, :, 1:-1], torch.ones((T, C, N, N, N - 1))
        )

    def test_get_half_interior_x(self) -> None:
        """Test get_half_interior method."""
        T, C, N, H = 1, 1, 8, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        value = torch.arange(0, N * N * N, dtype=dtype, device=device).reshape(
            1, 1, N, N, N
        )
        cell_field.interior = value

        # Test with offset [1, 0, 0] (+x, -y, -z)
        offset = np.array([1, 0, 0], dtype=np.uint32)
        half_interior = cell_field.get_half_interior(offset)

        halfN = N // 2
        expected = value[:, :, :-halfN:, :-halfN, halfN:]

        assert half_interior.shape == (T, C, halfN, halfN, halfN)
        assert torch.allclose(half_interior, expected)

    def test_get_half_interior_yz(self) -> None:
        """Test get_half_interior with different offsets."""
        T, C, N, H = 1, 2, 8, 1
        dtype = torch.float32
        device = torch.device("cpu")

        cell_field = CellField(T, C, N, H, dtype, device)
        value = torch.arange(0, N * N * N, dtype=dtype, device=device).reshape(
            1, 1, N, N, N
        )
        cell_field.interior = value

        # Test with offset [0, 1, 1] (-x, +y, +z)
        offset = np.array([0, 1, 1], dtype=np.uint32)
        half_interior = cell_field.get_half_interior(offset)

        halfN = N // 2
        expected = value[:, :, halfN:, halfN:, :-halfN]
        assert half_interior.shape == (T, C, halfN, halfN, halfN)
        assert torch.allclose(half_interior, expected)
