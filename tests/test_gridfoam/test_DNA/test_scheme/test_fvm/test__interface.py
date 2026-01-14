"""Tests for FVM operator interface."""

import pytest

from gridfoam.DNA.scheme.fvm._interface import IFVMOperator


def test_ifvm_operator_is_abstract():
    """Test that IFVMOperator is abstract."""
    with pytest.raises(TypeError):
        IFVMOperator()


def test_ifvm_operator_has_build_method():
    """Test that IFVMOperator has build method."""
    assert hasattr(IFVMOperator, "build")
    assert IFVMOperator.build.__isabstractmethod__
