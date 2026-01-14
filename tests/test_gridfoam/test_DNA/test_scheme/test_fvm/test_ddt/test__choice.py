"""Tests for FVM ddt scheme choice."""

import pytest

from gridfoam.DNA.scheme.fvm.ddt._choice import FVMDdtSchemeChoice


def test_fvm_ddt_scheme_choice_enum():
    """Test FVMDdtSchemeChoice enum values."""
    assert FVMDdtSchemeChoice.EULER == "Euler"
