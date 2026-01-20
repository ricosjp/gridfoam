"""Tests for FVC operator interface."""

from gridfoam.DNA.scheme.fvc._interface import IFVCOperator


def test_ifvc_operator_has_apply_method():
    """Test that IFVCOperator has apply method."""
    assert hasattr(IFVCOperator, "apply")
    assert IFVCOperator.apply.__isabstractmethod__
