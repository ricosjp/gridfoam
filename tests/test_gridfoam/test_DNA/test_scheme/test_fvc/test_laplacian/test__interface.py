"""Tests for FVC laplacian operator interface."""

import pytest

from gridfoam.DNA.scheme.fvc.laplacian._interface import IFVCLaplacianOperator


def test_ifvc_laplacian_operator_is_abstract():
    """Test that IFVCLaplacianOperator is abstract."""
    with pytest.raises(TypeError):
        IFVCLaplacianOperator.apply(None, None, None)


def test_ifvc_laplacian_operator_inherits_from_ifvc_operator():
    """Test that IFVCLaplacianOperator inherits from IFVCOperator."""
    from gridfoam.DNA.scheme.fvc._interface import IFVCOperator

    assert issubclass(IFVCLaplacianOperator, IFVCOperator)
