"""Tests for CellField."""

import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from gridfoam.DNA.enum import Axis
from gridfoam.DNA.fielddata._cell import CellField


def test_cell_field_init():
    """Test CellField initialization."""
    field = CellField(
        T=1, C=3, N=8, H=2, dtype=torch.float32, device=torch.device("cpu")
    )
    assert field.T == 1
    assert field.C == 3
    assert field.N == 8
    assert field.H == 2
    assert field.raw.shape == (1, 3, 12, 12, 12)  # N + 2*H = 8 + 4 = 12
    assert field.interior.shape == (1, 3, 8, 8, 8)


def test_cell_field_interior_access():
    """Test CellField interior property access."""
    field = CellField(
        T=1, C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    field.interior[0] = torch.ones(1, 4, 4, 4)
    assert torch.allclose(field.interior[0], torch.ones(1, 4, 4, 4))


def test_cell_field_zeros_like():
    """Test CellField.zeros_like method."""
    field1 = CellField(
        T=2, C=3, N=8, H=2, dtype=torch.float64, device=torch.device("cpu")
    )
    field2 = CellField.zeros_like(field1)
    assert field2.T == field1.T
    assert field2.C == field1.C
    assert field2.N == field1.N
    assert field2.H == field1.H
    assert field2.raw.dtype == field1.raw.dtype
    assert field2.raw.device == field1.raw.device


def test_cell_field_add():
    """Test CellField addition."""
    field1 = CellField(
        T=1, C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    field2 = CellField(
        T=1, C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    field1.interior[0] = torch.ones(1, 4, 4, 4)
    field2.interior[0] = torch.ones(1, 4, 4, 4) * 2
    result = field1 + field2
    assert torch.allclose(result.interior[0], torch.ones(1, 4, 4, 4) * 3)


def test_cell_field_sub():
    """Test CellField subtraction."""
    field1 = CellField(
        T=1, C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    field2 = CellField(
        T=1, C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    field1.interior[0] = torch.ones(1, 4, 4, 4) * 3
    field2.interior[0] = torch.ones(1, 4, 4, 4) * 2
    result = field1 - field2
    assert torch.allclose(result.interior[0], torch.ones(1, 4, 4, 4))


def test_cell_field_mul():
    """Test CellField multiplication."""
    field1 = CellField(
        T=1, C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    field2 = CellField(
        T=1, C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    field1.interior[0] = torch.ones(1, 4, 4, 4) * 2
    field2.interior[0] = torch.ones(1, 4, 4, 4) * 3
    result = field1 * field2
    assert torch.allclose(result.interior[0], torch.ones(1, 4, 4, 4) * 6)


def test_cell_field_mul_scalar():
    """Test CellField multiplication with scalar."""
    field = CellField(
        T=1, C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    field.interior[0] = torch.ones(1, 4, 4, 4) * 2
    result = field * 3.0
    assert torch.allclose(result.interior[0], torch.ones(1, 4, 4, 4) * 6)


def test_cell_field_imul_scalar():
    """Test CellField in-place multiplication with scalar."""
    field = CellField(
        T=1, C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    field.interior[0] = torch.ones(1, 4, 4, 4) * 2
    field *= 3.0
    assert torch.allclose(field.interior[0], torch.ones(1, 4, 4, 4) * 6)


def test_cell_field_imul_unsupported():
    """Test CellField in-place multiplication with unsupported type."""
    field = CellField(
        T=1, C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    with pytest.raises(BeartypeCallHintParamViolation):
        field *= "invalid"


def test_cell_field_get_boundary_cell_along():
    """Test CellField get_boundary_cell_along method."""
    field = CellField(
        T=1, C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    field.interior[0] = torch.arange(1, 65).reshape(1, 4, 4, 4).float()
    boundary = field.get_boundary_cell_along(Axis.X, forward=True)
    assert boundary.shape == (1, 1, 4, 4)


def test_cell_field_set_boundary_cell_along():
    """Test CellField set_boundary_cell_along method."""
    field = CellField(
        T=1, C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    value = torch.ones(1, 1, 4, 4) * 5.0
    field.set_boundary_cell_along(Axis.X, forward=True, value=value)
    boundary = field.get_boundary_cell_along(Axis.X, forward=True)
    assert torch.allclose(boundary, value)


def test_cell_field_face_average():
    """Test CellField face_average method."""
    field = CellField(
        T=1, C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    field.interior[0] = torch.ones(1, 4, 4, 4)
    face_field = field.face_average()
    assert face_field.T == 1
    assert face_field.C == 1
    assert face_field.N == 4
    assert face_field.x.shape == (1, 1, 4, 4, 5)  # N+1 for face
    assert face_field.y.shape == (1, 1, 4, 5, 4)
    assert face_field.z.shape == (1, 1, 5, 4, 4)


def test_cell_field_face_harmonic_mean():
    """Test CellField face_harmonic_mean method."""
    field = CellField(
        T=1, C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    field.interior[0] = torch.ones(1, 4, 4, 4) * 2.0
    face_field = field.face_harmonic_mean()
    assert face_field.T == 1
    assert face_field.C == 1
    assert face_field.N == 4


def test_cell_field_face_grad():
    """Test CellField face_grad method."""
    field = CellField(
        T=1, C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    # Create a linear field: x coordinate
    for i in range(4):
        field.interior[0, 0, :, :, i] = float(i)
    dx = torch.tensor([1.0, 1.0, 1.0])
    grad_field = field.face_grad(dx)
    assert grad_field.T == 1
    assert grad_field.C == 1
    assert grad_field.N == 4
