"""Tests for linear solver interface."""

import pytest

from gridfoam.DNA.scheme.solver._interface import ILinearSolver


def test_ilinear_solver_is_abstract():
    """Test that ILinearSolver is abstract."""
    with pytest.raises(TypeError):
        ILinearSolver(None, None)


def test_ilinear_solver_has_required_fields():
    """Test that ILinearSolver has required_fields property."""
    assert hasattr(ILinearSolver, "required_fields")


def test_ilinear_solver_has_solve_method():
    """Test that ILinearSolver has solve method."""
    assert hasattr(ILinearSolver, "solve")
    assert ILinearSolver.solve.__isabstractmethod__
