"""Tests for FVM div operator interface."""

import pytest

from gridfoam.DNA.scheme.fvm.div._interface import IFVMDivOperator


def test_ifvm_div_operator_is_abstract():
    """Test that IFVMDivOperator is abstract."""
    with pytest.raises(TypeError):
        IFVMDivOperator(None, None)


def test_ifvm_div_operator_inherits_from_ifvm_operator():
    """Test that IFVMDivOperator inherits from IFVMOperator."""
    from gridfoam.DNA.scheme.fvm._interface import IFVMOperator

    assert issubclass(IFVMDivOperator, IFVMOperator)
