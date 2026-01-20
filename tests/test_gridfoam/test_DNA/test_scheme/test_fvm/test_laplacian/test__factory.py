"""Tests for FVM laplacian scheme factory."""

from unittest.mock import Mock

import pytest

from gridfoam.DNA.enum import FieldLayout, FieldRole
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.laplacian._choice import FVMLaplacianSchemeChoice
from gridfoam.DNA.scheme.fvm.laplacian._factory import FVMLaplacianSchemeConfig
from gridfoam.DNA.scheme.fvm.laplacian._interface import IFVMLaplacianOperator


def test_fvm_laplacian_scheme_config_create_linear():
    """Test FVMLaplacianSchemeConfig creates Linear operator."""
    config = FVMLaplacianSchemeConfig(choice=FVMLaplacianSchemeChoice.LINEAR)
    gamma_fm = FieldMeta(
        name="gamma",
        label="Gamma",
        layout=FieldLayout.CELL,
        role=FieldRole.AUXILIARY,
    )
    psi_fm = FieldMeta(
        name="psi",
        label="Psi",
        layout=FieldLayout.CELL,
        role=FieldRole.STATE,
    )
    operator = config.create_operator(gamma_fm, psi_fm)
    assert isinstance(operator, IFVMLaplacianOperator)


def test_fvm_laplacian_scheme_config_unknown_choice():
    """Test FVMLaplacianSchemeConfig raises error for unknown choice."""
    mock_choice = Mock(spec=FVMLaplacianSchemeChoice)
    mock_choice.name = "UNKNOWN"

    config = FVMLaplacianSchemeConfig(choice=mock_choice)
    gamma_fm = FieldMeta(
        name="gamma",
        label="Gamma",
        layout=FieldLayout.CELL,
        role=FieldRole.AUXILIARY,
    )
    psi_fm = FieldMeta(
        name="psi",
        label="Psi",
        layout=FieldLayout.CELL,
        role=FieldRole.STATE,
    )
    with pytest.raises(ValueError, match="Unknown laplacian scheme choice"):
        config.create_operator(gamma_fm, psi_fm)
