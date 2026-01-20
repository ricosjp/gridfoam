"""Tests for FVM laplacian linear operator."""

import pytest

from gridfoam.DNA.enum import FieldLayout
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.laplacian._linear import FVMLaplacianLinear


def test_fvm_laplacian_linear_init():
    """Test FVMLaplacianLinear initialization."""
    gamma_field = FieldMeta(
        name="gamma",
        label="Gamma",
        layout=FieldLayout.CELL,
    )
    psi_field = FieldMeta(
        name="psi",
        label="Psi",
        layout=FieldLayout.CELL,
    )
    operator = FVMLaplacianLinear(gamma_field, psi_field)
    assert operator._gamma_fm == gamma_field
    assert operator._psi_fm == psi_field


def test_fvm_laplacian_linear_init_errors():
    """Test FVMLaplacianLinear raises errors for invalid layouts."""
    gamma_field = FieldMeta(
        name="gamma",
        label="Gamma",
        layout=FieldLayout.FACE,  # Wrong layout
    )
    psi_field = FieldMeta(
        name="psi",
        label="Psi",
        layout=FieldLayout.CELL,
    )
    with pytest.raises(AssertionError):
        FVMLaplacianLinear(gamma_field, psi_field)
