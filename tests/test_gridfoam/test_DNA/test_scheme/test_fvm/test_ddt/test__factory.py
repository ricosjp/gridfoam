"""Tests for FVM ddt scheme factory."""

from unittest.mock import Mock

import pytest

from gridfoam.DNA.enum import FieldLayout, FieldRole
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.ddt._choice import FVMDdtSchemeChoice
from gridfoam.DNA.scheme.fvm.ddt._factory import FVMDdtSchemeConfig
from gridfoam.DNA.scheme.fvm.ddt._interface import IFVMDdtOperator


def test_fvm_ddt_scheme_config_create_euler():
    """Test FVMDdtSchemeConfig creates Euler operator."""
    config = FVMDdtSchemeConfig(choice=FVMDdtSchemeChoice.EULER)
    psi_fm = FieldMeta(
        name="test",
        label="Test",
        role=FieldRole.STATE,
        layout=FieldLayout.CELL,
        components=1,
    )
    operator = config.create_operator(psi_fm)
    assert isinstance(operator, IFVMDdtOperator)


def test_fvm_ddt_scheme_config_unknown_choice():
    """Test FVMDdtSchemeConfig raises error for unknown choice."""
    mock_choice = Mock(spec=FVMDdtSchemeChoice)
    mock_choice.name = "UNKNOWN"

    config = FVMDdtSchemeConfig(choice=mock_choice)
    psi_fm = FieldMeta(
        name="test",
        label="Test",
        role=FieldRole.STATE,
        layout=FieldLayout.CELL,
        components=1,
    )
    with pytest.raises(ValueError, match="Unknown ddt scheme choice"):
        config.create_operator(psi_fm)
