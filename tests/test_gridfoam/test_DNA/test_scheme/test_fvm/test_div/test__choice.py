"""Tests for FVM div scheme choice."""

import pytest

from gridfoam.DNA.scheme.fvm.div._choice import FVMDivSchemeChoice


def test_fvm_div_scheme_choice_enum():
    """Test FVMDivSchemeChoice enum values."""
    assert FVMDivSchemeChoice.UPWIND == "Upwind"
    assert FVMDivSchemeChoice.LIMITED_LINEAR == "LimitedLinear"
    assert FVMDivSchemeChoice.LIMITED_LINEAR_V == "LimitedLinearV"
