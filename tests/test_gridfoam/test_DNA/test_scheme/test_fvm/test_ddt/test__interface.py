"""Tests for FVM ddt operator interface."""

import pytest

from gridfoam.DNA.scheme.fvm.ddt._interface import IFVMDdtOperator


def test_ifvm_ddt_operator_is_abstract():
    """Test that IFVMDdtOperator is abstract."""
    with pytest.raises(TypeError):
        IFVMDdtOperator(None)


def test_ifvm_ddt_operator_inherits_from_ifvm_operator():
    """Test that IFVMDdtOperator inherits from IFVMOperator."""
    from gridfoam.DNA.scheme.fvm._interface import IFVMOperator

    assert issubclass(IFVMDdtOperator, IFVMOperator)
