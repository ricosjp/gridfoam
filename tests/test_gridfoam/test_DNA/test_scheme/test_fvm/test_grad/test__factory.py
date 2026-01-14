"""Tests for FVM grad scheme factory."""

import pytest

from gridfoam.DNA.enum import FieldLayout
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.grad._choice import FVMGradSchemeChoice
from gridfoam.DNA.scheme.fvm.grad._factory import FVMGradSchemeConfig
from gridfoam.DNA.scheme.fvm.grad._interface import IFVMGradOperator


def test_fvm_grad_scheme_config_create_linear():
    """Test FVMGradSchemeConfig creates Linear operator."""
    config = FVMGradSchemeConfig(choice=FVMGradSchemeChoice.LINEAR)
    psi_fm = FieldMeta(
        name="test",
        layout=FieldLayout.CELL,
        role=None,
        precision=None,
    )
    operator = config.create_operator(psi_fm)
    assert isinstance(operator, IFVMGradOperator)


def test_fvm_grad_scheme_config_unknown_choice():
    """Test FVMGradSchemeConfig raises error for unknown choice."""
    class MockChoice:
        name = "UNKNOWN"

    config = FVMGradSchemeConfig(choice=MockChoice())
    psi_fm = FieldMeta(
        name="test",
        layout=FieldLayout.CELL,
        role=None,
        precision=None,
    )
    with pytest.raises(ValueError, match="Unknown grad scheme choice"):
        config.create_operator(psi_fm)
