"""Tests for solver method choice."""

import pytest

from gridfoam.DNA.scheme.solver._choice import SolverMethodChoice


def test_solver_method_choice_enum():
    """Test SolverMethodChoice enum values."""
    assert SolverMethodChoice.CG == "CG"
    assert SolverMethodChoice.BICGSTAB == "BiCGSTAB"
