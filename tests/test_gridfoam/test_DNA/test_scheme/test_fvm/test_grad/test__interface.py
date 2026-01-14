"""Tests for FVM grad operator interface."""

import pytest

from gridfoam.DNA.scheme.fvm.grad._interface import IFVMGradOperator


def test_ifvm_grad_operator_is_abstract():
    """Test that IFVMGradOperator is abstract."""
    with pytest.raises(TypeError):
        IFVMGradOperator(None)


def test_ifvm_grad_operator_inherits_from_ifvm_operator():
    """Test that IFVMGradOperator inherits from IFVMOperator."""
    from gridfoam.DNA.scheme.fvm._interface import IFVMOperator

    assert issubclass(IFVMGradOperator, IFVMOperator)
