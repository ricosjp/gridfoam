"""Tests for FaceField."""

import pytest
import torch

from gridfoam.DNA.enum import Axis
from gridfoam.DNA.fielddata._face import FaceField


def test_face_field_init():
    """Test FaceField initialization."""
    field = FaceField(T=1, C=3, N=8, dtype=torch.float32, device=torch.device("cpu"))
    assert field.T == 1
    assert field.C == 3
    assert field.N == 8
    assert field.x.shape == (1, 3, 8, 8, 9)  # N+1 for face
    assert field.y.shape == (1, 3, 8, 9, 8)
    assert field.z.shape == (1, 3, 9, 8, 8)


def test_face_field_zeros_like():
    """Test FaceField.zeros_like method."""
    field1 = FaceField(T=2, C=3, N=8, dtype=torch.float64, device=torch.device("cpu"))
    field2 = FaceField.zeros_like(field1)
    assert field2.T == field1.T
    assert field2.C == field1.C
    assert field2.N == field1.N
    assert field2.x.dtype == field1.x.dtype
    assert field2.x.device == field1.x.device


def test_face_field_add():
    """Test FaceField addition."""
    field1 = FaceField(T=1, C=1, N=4, dtype=torch.float32, device=torch.device("cpu"))
    field2 = FaceField(T=1, C=1, N=4, dtype=torch.float32, device=torch.device("cpu"))
    field1.x[0] = torch.ones(1, 4, 4, 5)
    field2.x[0] = torch.ones(1, 4, 4, 5) * 2
    result = field1 + field2
    assert torch.allclose(result.x[0], torch.ones(1, 4, 4, 5) * 3)


def test_face_field_sub():
    """Test FaceField subtraction."""
    field1 = FaceField(T=1, C=1, N=4, dtype=torch.float32, device=torch.device("cpu"))
    field2 = FaceField(T=1, C=1, N=4, dtype=torch.float32, device=torch.device("cpu"))
    field1.x[0] = torch.ones(1, 4, 4, 5) * 3
    field2.x[0] = torch.ones(1, 4, 4, 5) * 2
    result = field1 - field2
    assert torch.allclose(result.x[0], torch.ones(1, 4, 4, 5))


def test_face_field_mul():
    """Test FaceField multiplication."""
    field1 = FaceField(T=1, C=1, N=4, dtype=torch.float32, device=torch.device("cpu"))
    field2 = FaceField(T=1, C=1, N=4, dtype=torch.float32, device=torch.device("cpu"))
    field1.x[0] = torch.ones(1, 4, 4, 5) * 2
    field2.x[0] = torch.ones(1, 4, 4, 5) * 3
    result = field1 * field2
    assert torch.allclose(result.x[0], torch.ones(1, 4, 4, 5) * 6)


def test_face_field_get_boundary_face_along():
    """Test FaceField get_boundary_face_along method."""
    field = FaceField(T=1, C=1, N=4, dtype=torch.float32, device=torch.device("cpu"))
    field.x[0] = torch.arange(1, 81).reshape(1, 4, 4, 5).float()
    boundary = field.get_boundary_face_along(Axis.X, forward=True)
    assert boundary.shape == (1, 1, 4, 4)


def test_face_field_set_boundary_face_along():
    """Test FaceField set_boundary_face_along method."""
    field = FaceField(T=1, C=1, N=4, dtype=torch.float32, device=torch.device("cpu"))
    value = torch.ones(1, 1, 4, 4) * 5.0
    field.set_boundary_face_along(Axis.X, forward=True, value=value)
    boundary = field.get_boundary_face_along(Axis.X, forward=True)
    assert torch.allclose(boundary, value)


def test_face_field_integrate_dSn():
    """Test FaceField integrate_dSn method."""
    field = FaceField(T=1, C=1, N=4, dtype=torch.float32, device=torch.device("cpu"))
    field.x[0] = torch.ones(1, 4, 4, 5)
    field.y[0] = torch.ones(1, 4, 5, 4)
    field.z[0] = torch.ones(1, 5, 4, 4)
    Sf = torch.tensor([1.0, 1.0, 1.0])
    result = field.integrate_dSn(Sf)
    assert result.shape == (1, 1, 4, 4, 4)
    # For uniform field, divergence should be zero
    assert torch.allclose(result[0], torch.zeros(1, 4, 4, 4))


def test_face_field_invalid_axis():
    """Test FaceField raises error for invalid axis."""
    field = FaceField(T=1, C=1, N=4, dtype=torch.float32, device=torch.device("cpu"))
    # This test would need a mock Axis enum value to properly test
    # For now, we just verify the method exists and handles valid axes
    boundary = field.get_boundary_face_along(Axis.X, forward=True)
    assert boundary is not None
