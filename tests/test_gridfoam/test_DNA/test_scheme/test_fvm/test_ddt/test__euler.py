"""Tests for FVM ddt Euler operator."""

import pytest

from gridfoam.DNA.enum import FieldLayout
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.ddt._euler import FVMDdtEuler


def test_fvm_ddt_euler_init():
    """Test FVMDdtEuler initialization."""
    field = FieldMeta(
        name="test",
        label="Test",
        layout=FieldLayout.CELL,
    )
    operator = FVMDdtEuler(field)
    assert operator._psi_fm == field


def test_fvm_ddt_euler_init_face_error():
    """Test FVMDdtEuler raises error for face-centered field."""
    field = FieldMeta(
        name="test",
        label="Test",
        layout=FieldLayout.FACE,
    )
    with pytest.raises(AssertionError):
        FVMDdtEuler(field)
