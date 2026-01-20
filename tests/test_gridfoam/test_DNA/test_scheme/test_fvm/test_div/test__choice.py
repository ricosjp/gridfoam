"""Tests for FVM div scheme choice."""


from gridfoam.DNA.scheme.fvm.div._choice import FVMDivSchemeChoice


def test_fvm_div_scheme_choice_enum():
    """Test FVMDivSchemeChoice enum values."""
    assert FVMDivSchemeChoice.UPWIND.value == "Upwind"
    assert FVMDivSchemeChoice.LIMITED_LINEAR.value == "LimitedLinear"
    assert FVMDivSchemeChoice.LIMITED_LINEAR_V.value == "LimitedLinearV"
