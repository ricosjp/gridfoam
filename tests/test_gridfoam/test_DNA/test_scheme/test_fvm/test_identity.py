"""Tests for FVM identity operator."""

import pytest

from gridfoam.DNA.enum import FieldLayout
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.identity import FVMIdentity


def test_fvm_identity_init():
    """Test FVMIdentity initialization."""
    field = FieldMeta(
        name="test",
        label="Test",
        layout=FieldLayout.CELL,
    )
    operator = FVMIdentity(field)
    assert operator._psi_fm == field


def test_fvm_identity_init_face_error():
    """Test FVMIdentity raises error for face-centered field."""
    field = FieldMeta(
        name="test",
        label="Test",
        layout=FieldLayout.FACE,
    )
    with pytest.raises(AssertionError):
        FVMIdentity(field)
