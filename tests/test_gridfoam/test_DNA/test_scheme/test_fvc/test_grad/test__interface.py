"""Tests for FVC grad operator interface."""

import pytest

from gridfoam.DNA.scheme.fvc.grad._interface import IFVCGradOperator


def test_ifvc_grad_operator_is_abstract():
    """Test that IFVCGradOperator is abstract."""
    with pytest.raises(TypeError):
        IFVCGradOperator.apply(None, None)


def test_ifvc_grad_operator_inherits_from_ifvc_operator():
    """Test that IFVCGradOperator inherits from IFVCOperator."""
    from gridfoam.DNA.scheme.fvc._interface import IFVCOperator

    assert issubclass(IFVCGradOperator, IFVCOperator)
