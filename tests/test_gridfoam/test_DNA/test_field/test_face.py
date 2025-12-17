from __future__ import annotations

import torch

from gridfoam.DNA.enum import Axis
from gridfoam.DNA.field import FaceField


class TestFaceField:
    """Test suite for FaceField class."""

    def test_init(self) -> None:
        """Test FaceField initialization."""
        T, C, N = 2, 3, 4
        dtype = torch.float32
        device = torch.device("cpu")

        face_field = FaceField(T, C, N, dtype, device)

        assert face_field.T == T
        assert face_field.C == C
        assert face_field.N == N
        assert face_field.x.shape == (T, C, N, N, N + 1)
        assert face_field.y.shape == (T, C, N, N + 1, N)
        assert face_field.z.shape == (T, C, N + 1, N, N)
        assert face_field.x.dtype == dtype
        assert face_field.x.device == device
        assert torch.allclose(face_field.x, torch.zeros_like(face_field.x))
        assert torch.allclose(face_field.y, torch.zeros_like(face_field.y))
        assert torch.allclose(face_field.z, torch.zeros_like(face_field.z))

    def test_properties(self) -> None:
        """Test FaceField properties."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        face_field = FaceField(T, C, N, dtype, device)

        # Test property getters
        assert face_field.T == T
        assert face_field.C == C
        assert face_field.N == N

        # Test property setters
        new_x = torch.ones((T, C, N, N, N + 1), dtype=dtype, device=device)
        new_y = torch.ones((T, C, N, N + 1, N), dtype=dtype, device=device)
        new_z = torch.ones((T, C, N + 1, N, N), dtype=dtype, device=device)

        face_field.x = new_x
        face_field.y = new_y
        face_field.z = new_z

        assert torch.allclose(face_field.x, new_x)
        assert torch.allclose(face_field.y, new_y)
        assert torch.allclose(face_field.z, new_z)

    def test_zeros_like(self) -> None:
        """Test zeros_like class method."""
        T, C, N = 2, 3, 4
        dtype = torch.float32
        device = torch.device("cpu")

        original = FaceField(T, C, N, dtype, device)
        original.x.fill_(1.0)
        original.y.fill_(2.0)
        original.z.fill_(3.0)

        zeros_field = FaceField.zeros_like(original)

        assert zeros_field.T == original.T
        assert zeros_field.C == original.C
        assert zeros_field.N == original.N
        assert zeros_field.x.shape == original.x.shape
        assert zeros_field.y.shape == original.y.shape
        assert zeros_field.z.shape == original.z.shape
        assert zeros_field.x.dtype == original.x.dtype
        assert zeros_field.x.device == original.x.device
        assert torch.allclose(zeros_field.x, torch.zeros_like(zeros_field.x))
        assert torch.allclose(zeros_field.y, torch.zeros_like(zeros_field.y))
        assert torch.allclose(zeros_field.z, torch.zeros_like(zeros_field.z))

    def test_add(self) -> None:
        """Test addition operator."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        field1 = FaceField(T, C, N, dtype, device)
        field2 = FaceField(T, C, N, dtype, device)

        field1.x.fill_(1.0)
        field1.y.fill_(2.0)
        field1.z.fill_(3.0)

        field2.x.fill_(4.0)
        field2.y.fill_(5.0)
        field2.z.fill_(6.0)

        result = field1 + field2

        assert torch.allclose(result.x, torch.full_like(result.x, 5.0))
        assert torch.allclose(result.y, torch.full_like(result.y, 7.0))
        assert torch.allclose(result.z, torch.full_like(result.z, 9.0))

    def test_sub(self) -> None:
        """Test subtraction operator."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        field1 = FaceField(T, C, N, dtype, device)
        field2 = FaceField(T, C, N, dtype, device)

        field1.x.fill_(5.0)
        field1.y.fill_(7.0)
        field1.z.fill_(9.0)

        field2.x.fill_(1.0)
        field2.y.fill_(2.0)
        field2.z.fill_(3.0)

        result = field1 - field2

        assert torch.allclose(result.x, torch.full_like(result.x, 4.0))
        assert torch.allclose(result.y, torch.full_like(result.y, 5.0))
        assert torch.allclose(result.z, torch.full_like(result.z, 6.0))

    def test_mul(self) -> None:
        """Test multiplication operator."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        field1 = FaceField(T, C, N, dtype, device)
        field2 = FaceField(T, C, N, dtype, device)

        field1.x.fill_(2.0)
        field1.y.fill_(3.0)
        field1.z.fill_(4.0)

        field2.x.fill_(5.0)
        field2.y.fill_(6.0)
        field2.z.fill_(7.0)

        result = field1 * field2

        assert torch.allclose(result.x, torch.full_like(result.x, 10.0))
        assert torch.allclose(result.y, torch.full_like(result.y, 18.0))
        assert torch.allclose(result.z, torch.full_like(result.z, 28.0))

    def test_get_boundary_face_along_x_forward(self) -> None:
        """Test getting boundary face along X axis (forward)."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        face_field = FaceField(T, C, N, dtype, device)
        face_field.x[:, :, :, :, N] = 1.0  # Set forward boundary

        boundary = face_field.get_boundary_face_along(Axis.X, forward=True)

        assert boundary.shape == (T, C, N, N)
        assert torch.allclose(boundary, torch.ones((T, C, N, N)))

    def test_get_boundary_face_along_x_backward(self) -> None:
        """Test getting boundary face along X axis (backward)."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        face_field = FaceField(T, C, N, dtype, device)
        face_field.x[:, :, :, :, 0] = 2.0  # Set backward boundary

        boundary = face_field.get_boundary_face_along(Axis.X, forward=False)

        assert boundary.shape == (T, C, N, N)
        assert torch.allclose(boundary, torch.full((T, C, N, N), 2.0))

    def test_get_boundary_face_along_y_forward(self) -> None:
        """Test getting boundary face along Y axis (forward)."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        face_field = FaceField(T, C, N, dtype, device)
        face_field.y[:, :, :, N, :] = 3.0  # Set forward boundary

        boundary = face_field.get_boundary_face_along(Axis.Y, forward=True)

        assert boundary.shape == (T, C, N, N)
        assert torch.allclose(boundary, torch.full((T, C, N, N), 3.0))

    def test_get_boundary_face_along_y_backward(self) -> None:
        """Test getting boundary face along Y axis (backward)."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        face_field = FaceField(T, C, N, dtype, device)
        face_field.y[:, :, :, 0, :] = 4.0  # Set backward boundary

        boundary = face_field.get_boundary_face_along(Axis.Y, forward=False)

        assert boundary.shape == (T, C, N, N)
        assert torch.allclose(boundary, torch.full((T, C, N, N), 4.0))

    def test_get_boundary_face_along_z_forward(self) -> None:
        """Test getting boundary face along Z axis (forward)."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        face_field = FaceField(T, C, N, dtype, device)
        face_field.z[:, :, N, :, :] = 5.0  # Set forward boundary

        boundary = face_field.get_boundary_face_along(Axis.Z, forward=True)

        assert boundary.shape == (T, C, N, N)
        assert torch.allclose(boundary, torch.full((T, C, N, N), 5.0))

    def test_get_boundary_face_along_z_backward(self) -> None:
        """Test getting boundary face along Z axis (backward)."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        face_field = FaceField(T, C, N, dtype, device)
        face_field.z[:, :, 0, :, :] = 6.0  # Set backward boundary

        boundary = face_field.get_boundary_face_along(Axis.Z, forward=False)

        assert boundary.shape == (T, C, N, N)
        assert torch.allclose(boundary, torch.full((T, C, N, N), 6.0))

    def test_set_boundary_face_along_x_forward(self) -> None:
        """Test setting boundary face along X axis (forward)."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        face_field = FaceField(T, C, N, dtype, device)
        value = torch.ones((T, C, N, N), dtype=dtype, device=device) * 7.0

        face_field.set_boundary_face_along(Axis.X, forward=True, value=value)

        boundary = face_field.get_boundary_face_along(Axis.X, forward=True)
        assert torch.allclose(boundary, value)

    def test_set_boundary_face_along_x_backward(self) -> None:
        """Test setting boundary face along X axis (backward)."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        face_field = FaceField(T, C, N, dtype, device)
        value = torch.ones((T, C, N, N), dtype=dtype, device=device) * 8.0

        face_field.set_boundary_face_along(Axis.X, forward=False, value=value)

        boundary = face_field.get_boundary_face_along(Axis.X, forward=False)
        assert torch.allclose(boundary, value)

    def test_set_boundary_face_along_y_forward(self) -> None:
        """Test setting boundary face along Y axis (forward)."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        face_field = FaceField(T, C, N, dtype, device)
        value = torch.ones((T, C, N, N), dtype=dtype, device=device) * 9.0

        face_field.set_boundary_face_along(Axis.Y, forward=True, value=value)

        boundary = face_field.get_boundary_face_along(Axis.Y, forward=True)
        assert torch.allclose(boundary, value)

    def test_set_boundary_face_along_y_backward(self) -> None:
        """Test setting boundary face along Y axis (backward)."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        face_field = FaceField(T, C, N, dtype, device)
        value = torch.ones((T, C, N, N), dtype=dtype, device=device) * 10.0

        face_field.set_boundary_face_along(Axis.Y, forward=False, value=value)

        boundary = face_field.get_boundary_face_along(Axis.Y, forward=False)
        assert torch.allclose(boundary, value)

    def test_set_boundary_face_along_z_forward(self) -> None:
        """Test setting boundary face along Z axis (forward)."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        face_field = FaceField(T, C, N, dtype, device)
        value = torch.ones((T, C, N, N), dtype=dtype, device=device) * 11.0

        face_field.set_boundary_face_along(Axis.Z, forward=True, value=value)

        boundary = face_field.get_boundary_face_along(Axis.Z, forward=True)
        assert torch.allclose(boundary, value)

    def test_set_boundary_face_along_z_backward(self) -> None:
        """Test setting boundary face along Z axis (backward)."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        face_field = FaceField(T, C, N, dtype, device)
        value = torch.ones((T, C, N, N), dtype=dtype, device=device) * 12.0

        face_field.set_boundary_face_along(Axis.Z, forward=False, value=value)

        boundary = face_field.get_boundary_face_along(Axis.Z, forward=False)
        assert torch.allclose(boundary, value)

    def test_get_face_values_around_cell_x_forward(self) -> None:
        """Test getting face values around cell along X axis (forward)."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        face_field = FaceField(T, C, N, dtype, device)
        face_field.x[:, :, :, :, 1:] = 1.0  # Set forward faces

        values = face_field.get_face_values_around_cell(Axis.X, forward=True)

        assert values.shape == (T, C, N, N, N)
        assert torch.allclose(values, torch.ones((T, C, N, N, N)))

    def test_get_face_values_around_cell_x_backward(self) -> None:
        """Test getting face values around cell along X axis (backward)."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        face_field = FaceField(T, C, N, dtype, device)
        face_field.x[:, :, :, :, :-1] = 2.0  # Set backward faces

        values = face_field.get_face_values_around_cell(Axis.X, forward=False)

        assert values.shape == (T, C, N, N, N)
        assert torch.allclose(values, torch.full((T, C, N, N, N), 2.0))

    def test_get_face_values_around_cell_y_forward(self) -> None:
        """Test getting face values around cell along Y axis (forward)."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        face_field = FaceField(T, C, N, dtype, device)
        face_field.y[:, :, :, 1:, :] = 3.0  # Set forward faces

        values = face_field.get_face_values_around_cell(Axis.Y, forward=True)

        assert values.shape == (T, C, N, N, N)
        assert torch.allclose(values, torch.full((T, C, N, N, N), 3.0))

    def test_get_face_values_around_cell_y_backward(self) -> None:
        """Test getting face values around cell along Y axis (backward)."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        face_field = FaceField(T, C, N, dtype, device)
        face_field.y[:, :, :, :-1, :] = 4.0  # Set backward faces

        values = face_field.get_face_values_around_cell(Axis.Y, forward=False)

        assert values.shape == (T, C, N, N, N)
        assert torch.allclose(values, torch.full((T, C, N, N, N), 4.0))

    def test_get_face_values_around_cell_z_forward(self) -> None:
        """Test getting face values around cell along Z axis (forward)."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        face_field = FaceField(T, C, N, dtype, device)
        face_field.z[:, :, 1:, :, :] = 5.0  # Set forward faces

        values = face_field.get_face_values_around_cell(Axis.Z, forward=True)

        assert values.shape == (T, C, N, N, N)
        assert torch.allclose(values, torch.full((T, C, N, N, N), 5.0))

    def test_get_face_values_around_cell_z_backward(self) -> None:
        """Test getting face values around cell along Z axis (backward)."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        face_field = FaceField(T, C, N, dtype, device)
        face_field.z[:, :, :-1, :, :] = 6.0  # Set backward faces

        values = face_field.get_face_values_around_cell(Axis.Z, forward=False)

        assert values.shape == (T, C, N, N, N)
        assert torch.allclose(values, torch.full((T, C, N, N, N), 6.0))

    def test_integrate_dSn(self) -> None:
        """Test integrate_dSn method."""
        T, C, N = 1, 2, 3
        dtype = torch.float32
        device = torch.device("cpu")

        face_field = FaceField(T, C, N, dtype, device)

        # Set face values
        x_boundary_value = torch.ones((T, C, N, N), dtype=dtype, device=device)
        face_field.set_boundary_face_along(Axis.X, forward=True, value=x_boundary_value)
        face_field.set_boundary_face_along(Axis.X, forward=False, value=x_boundary_value)

        Sf = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)

        result = face_field.integrate_dSn(Sf)

        assert result.shape == (T, C, N, N, N)
        expected = torch.zeros((T, C, N, N, N), dtype=dtype, device=device)
        expected[:, :, :, :, 0] = -1.0
        expected[:, :, :, :, -1] = 1.0
        assert torch.allclose(result, expected)


