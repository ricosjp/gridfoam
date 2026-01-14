"""Tests for FVC div operator interface."""

import pytest

from gridfoam.DNA.scheme.fvc.div._interface import IFVCDivOperator


def test_ifvc_div_operator_is_abstract():
    """Test that IFVCDivOperator is abstract."""
    with pytest.raises(TypeError):
        IFVCDivOperator.apply(None, None)


def test_ifvc_div_operator_inherits_from_ifvc_operator():
    """Test that IFVCDivOperator inherits from IFVCOperator."""
    from gridfoam.DNA.scheme.fvc._interface import IFVCOperator

    assert issubclass(IFVCDivOperator, IFVCOperator)
