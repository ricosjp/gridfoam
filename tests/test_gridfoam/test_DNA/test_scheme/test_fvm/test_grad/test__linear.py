"""Tests for FVM grad linear operator."""

import pytest

from gridfoam.DNA.enum import FieldLayout
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.grad._linear import FVMGradLinear


def test_fvm_grad_linear_init():
    """Test FVMGradLinear initialization."""
    field = FieldMeta(
        name="test",
        label="Test",
        layout=FieldLayout.CELL,
    )
    operator = FVMGradLinear(field)
    assert operator._psi_fm == field


def test_fvm_grad_linear_init_face_error():
    """Test FVMGradLinear raises error for face-centered field."""
    field = FieldMeta(
        name="test",
        label="Test",
        layout=FieldLayout.FACE,
    )
    with pytest.raises(AssertionError):
        FVMGradLinear(field)
