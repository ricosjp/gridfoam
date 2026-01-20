"""Tests for FVM grad scheme choice."""

from gridfoam.DNA.scheme.fvm.grad._choice import FVMGradSchemeChoice


def test_fvm_grad_scheme_choice_enum():
    """Test FVMGradSchemeChoice enum values."""
    assert FVMGradSchemeChoice.LINEAR.value == "GaussLinear"
