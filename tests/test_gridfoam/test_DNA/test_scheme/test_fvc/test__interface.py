"""Tests for FVC operator interface."""

import pytest

from gridfoam.DNA.scheme.fvc._interface import IFVCOperator


def test_ifvc_operator_is_abstract():
    """Test that IFVCOperator is abstract."""
    with pytest.raises(TypeError):
        IFVCOperator.apply()


def test_ifvc_operator_has_apply_method():
    """Test that IFVCOperator has apply method."""
    assert hasattr(IFVCOperator, "apply")
    assert IFVCOperator.apply.__isabstractmethod__
