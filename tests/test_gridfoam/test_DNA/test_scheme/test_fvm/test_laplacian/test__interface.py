"""Tests for FVM laplacian operator interface."""

import pytest

from gridfoam.DNA.scheme.fvm.laplacian._interface import IFVMLaplacianOperator


def test_ifvm_laplacian_operator_is_abstract():
    """Test that IFVMLaplacianOperator is abstract."""
    with pytest.raises(TypeError):
        IFVMLaplacianOperator(None, None)


def test_ifvm_laplacian_operator_inherits_from_ifvm_operator():
    """Test that IFVMLaplacianOperator inherits from IFVMOperator."""
    from gridfoam.DNA.scheme.fvm._interface import IFVMOperator

    assert issubclass(IFVMLaplacianOperator, IFVMOperator)
