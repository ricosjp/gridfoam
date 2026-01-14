"""Tests for FVM laplacian scheme factory."""

import pytest

from gridfoam.DNA.enum import FieldLayout
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.laplacian._choice import FVMLaplacianSchemeChoice
from gridfoam.DNA.scheme.fvm.laplacian._factory import FVMLaplacianSchemeConfig
from gridfoam.DNA.scheme.fvm.laplacian._interface import IFVMLaplacianOperator


def test_fvm_laplacian_scheme_config_create_linear():
    """Test FVMLaplacianSchemeConfig creates Linear operator."""
    config = FVMLaplacianSchemeConfig(choice=FVMLaplacianSchemeChoice.LINEAR)
    gamma_fm = FieldMeta(
        name="gamma",
        layout=FieldLayout.CELL,
        role=None,
        precision=None,
    )
    psi_fm = FieldMeta(
        name="psi",
        layout=FieldLayout.CELL,
        role=None,
        precision=None,
    )
    operator = config.create_operator(gamma_fm, psi_fm)
    assert isinstance(operator, IFVMLaplacianOperator)


def test_fvm_laplacian_scheme_config_unknown_choice():
    """Test FVMLaplacianSchemeConfig raises error for unknown choice."""
    class MockChoice:
        name = "UNKNOWN"

    config = FVMLaplacianSchemeConfig(choice=MockChoice())
    gamma_fm = FieldMeta(
        name="gamma",
        layout=FieldLayout.CELL,
        role=None,
        precision=None,
    )
    psi_fm = FieldMeta(
        name="psi",
        layout=FieldLayout.CELL,
        role=None,
        precision=None,
    )
    with pytest.raises(ValueError, match="Unknown laplacian scheme choice"):
        config.create_operator(gamma_fm, psi_fm)
