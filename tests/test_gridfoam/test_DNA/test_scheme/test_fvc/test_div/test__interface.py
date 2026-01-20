"""Tests for FVC div operator interface."""


from gridfoam.DNA.scheme.fvc.div._interface import IFVCDivOperator


def test_ifvc_div_operator_inherits_from_ifvc_operator():
    """Test that IFVCDivOperator inherits from IFVCOperator."""
    from gridfoam.DNA.scheme.fvc._interface import IFVCOperator

    assert issubclass(IFVCDivOperator, IFVCOperator)
