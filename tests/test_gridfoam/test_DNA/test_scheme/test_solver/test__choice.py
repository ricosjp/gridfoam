"""Tests for solver method choice."""


from gridfoam.DNA.scheme.solver._choice import SolverMethodChoice


def test_solver_method_choice_enum():
    """Test SolverMethodChoice enum values."""
    assert SolverMethodChoice.CG.value == "CG"
    assert SolverMethodChoice.BICGSTAB.value == "BiCGSTAB"
