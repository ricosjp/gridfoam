"""Tests for FVM grad scheme factory."""

from unittest.mock import Mock

import pytest

from gridfoam.DNA.enum import FieldLayout, FieldRole
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.grad._choice import FVMGradSchemeChoice
from gridfoam.DNA.scheme.fvm.grad._factory import FVMGradSchemeConfig
from gridfoam.DNA.scheme.fvm.grad._interface import IFVMGradOperator


def test_fvm_grad_scheme_config_create_linear():
    """Test FVMGradSchemeConfig creates Linear operator."""
    config = FVMGradSchemeConfig(choice=FVMGradSchemeChoice.LINEAR)
    psi_fm = FieldMeta(
        name="test",
        label="Test",
        layout=FieldLayout.CELL,
        role=FieldRole.STATE,
    )
    operator = config.create_operator(psi_fm)
    assert isinstance(operator, IFVMGradOperator)


def test_fvm_grad_scheme_config_unknown_choice():
    """Test FVMGradSchemeConfig raises error for unknown choice."""
    mock_choice = Mock(spec=FVMGradSchemeChoice)
    mock_choice.name = "UNKNOWN"

    config = FVMGradSchemeConfig(choice=mock_choice)
    psi_fm = FieldMeta(
        name="test",
        label="Test",
        role=FieldRole.STATE,
        layout=FieldLayout.CELL,
    )
    with pytest.raises(ValueError, match="Unknown grad scheme choice"):
        config.create_operator(psi_fm)
