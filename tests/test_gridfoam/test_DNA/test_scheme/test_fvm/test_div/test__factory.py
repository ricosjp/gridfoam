"""Tests for FVM div scheme factory."""

from unittest.mock import Mock

import pytest

from gridfoam.DNA.enum import FieldLayout, FieldRole
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.div._choice import FVMDivSchemeChoice
from gridfoam.DNA.scheme.fvm.div._factory import FVMDivSchemeConfig
from gridfoam.DNA.scheme.fvm.div._interface import IFVMDivOperator


def test_fvm_div_scheme_config_create_upwind():
    """Test FVMDivSchemeConfig creates Upwind operator."""
    config = FVMDivSchemeConfig(choice=FVMDivSchemeChoice.UPWIND)
    phi_fm = FieldMeta(
        name="phi",
        label="phi",
        layout=FieldLayout.FACE,
        role=FieldRole.AUXILIARY,
    )
    psi_fm = FieldMeta(
        name="psi",
        label="psi",
        layout=FieldLayout.CELL,
        role=FieldRole.STATE,
    )
    operator = config.create_operator(phi_fm, psi_fm)
    assert isinstance(operator, IFVMDivOperator)


def test_fvm_div_scheme_config_create_limited_linear():
    """Test FVMDivSchemeConfig creates LimitedLinear operator."""
    config = FVMDivSchemeConfig(choice=FVMDivSchemeChoice.LIMITED_LINEAR)
    phi_fm = FieldMeta(
        name="phi",
        label="phi",
        layout=FieldLayout.FACE,
        role=FieldRole.AUXILIARY,
    )
    psi_fm = FieldMeta(
        name="psi",
        label="psi",
        layout=FieldLayout.CELL,
        role=FieldRole.STATE,
    )
    operator = config.create_operator(phi_fm, psi_fm)
    assert isinstance(operator, IFVMDivOperator)


def test_fvm_div_scheme_config_create_limited_linear_v():
    """Test FVMDivSchemeConfig creates LimitedLinearV operator."""
    config = FVMDivSchemeConfig(choice=FVMDivSchemeChoice.LIMITED_LINEAR_V)
    phi_fm = FieldMeta(
        name="phi",
        label="phi",
        layout=FieldLayout.FACE,
        role=FieldRole.AUXILIARY,
    )
    psi_fm = FieldMeta(
        name="psi",
        label="psi",
        layout=FieldLayout.CELL,
        role=FieldRole.STATE,
    )
    operator = config.create_operator(phi_fm, psi_fm)
    assert isinstance(operator, IFVMDivOperator)


def test_fvm_div_scheme_config_unknown_choice():
    """Test FVMDivSchemeConfig raises error for unknown choice."""
    mock_choice = Mock(spec=FVMDivSchemeChoice)
    mock_choice.name = "UNKNOWN"

    config = FVMDivSchemeConfig(choice=mock_choice)
    phi_fm = FieldMeta(
        name="phi",
        label="phi",
        layout=FieldLayout.FACE,
        role=FieldRole.AUXILIARY,
    )
    psi_fm = FieldMeta(
        name="psi",
        label="psi",
        layout=FieldLayout.CELL,
        role=FieldRole.STATE,
    )
    with pytest.raises(ValueError, match="Unknown div scheme choice"):
        config.create_operator(phi_fm, psi_fm)
