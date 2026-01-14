"""Tests for FVM ddt scheme factory."""

import pytest
from unittest.mock import MagicMock

from gridfoam.DNA.enum import FieldLayout
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.ddt._choice import FVMDdtSchemeChoice
from gridfoam.DNA.scheme.fvm.ddt._factory import FVMDdtSchemeConfig
from gridfoam.DNA.scheme.fvm.ddt._interface import IFVMDdtOperator


def test_fvm_ddt_scheme_config_create_euler():
    """Test FVMDdtSchemeConfig creates Euler operator."""
    config = FVMDdtSchemeConfig(choice=FVMDdtSchemeChoice.EULER)
    psi_fm = FieldMeta(
        name="test",
        layout=FieldLayout.CELL,
        role=None,
        precision=None,
    )
    operator = config.create_operator(psi_fm)
    assert isinstance(operator, IFVMDdtOperator)


def test_fvm_ddt_scheme_config_unknown_choice():
    """Test FVMDdtSchemeConfig raises error for unknown choice."""
    # Create a mock enum value
    class MockChoice:
        name = "UNKNOWN"

    config = FVMDdtSchemeConfig(choice=MockChoice())
    psi_fm = FieldMeta(
        name="test",
        layout=FieldLayout.CELL,
        role=None,
        precision=None,
    )
    with pytest.raises(ValueError, match="Unknown ddt scheme choice"):
        config.create_operator(psi_fm)
