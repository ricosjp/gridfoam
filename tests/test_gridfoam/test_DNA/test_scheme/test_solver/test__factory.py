"""Tests for solver factory."""

from gridfoam.DNA.config import SolverChoice
from gridfoam.DNA.enum import FieldLayout, FieldRole, NormType
from gridfoam.DNA.meta.equation import EquationMeta
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.solver._choice import SolverMethodChoice
from gridfoam.DNA.scheme.solver._factory import SolverFactory
from gridfoam.DNA.scheme.solver._interface import ILinearSolver


def test_solver_factory_create_cg():
    """Test SolverFactory creates CG solver."""
    solver_choice = SolverChoice(
        method=SolverMethodChoice.CG,
        tolerance=1e-6,
        rel_tolerance=1e-6,
        max_iter=1000,
        norm_type=NormType.L_2,
    )
    eq_meta = EquationMeta(
        name="test",
        target_field=FieldMeta(
            name="test_field",
            label="Test Field",
            layout=FieldLayout.CELL,
            role=FieldRole.STATE,
        ),
        boundary_conditions=[],
        ast_root=FieldMeta(
            name="test_field",
            label="Test Field",
            layout=FieldLayout.CELL,
            role=FieldRole.STATE,
        ),
    )
    solver = SolverFactory.create(solver_choice, eq_meta)
    assert isinstance(solver, ILinearSolver)


def test_solver_factory_create_bicgstab():
    """Test SolverFactory creates BiCGSTAB solver."""
    solver_choice = SolverChoice(
        method=SolverMethodChoice.BICGSTAB,
        tolerance=1e-6,
        rel_tolerance=1e-6,
        max_iter=1000,
        norm_type=NormType.L_2,
    )
    eq_meta = EquationMeta(
        name="test",
        target_field=FieldMeta(
            name="test_field",
            label="Test Field",
            layout=FieldLayout.CELL,
            role=FieldRole.STATE,
        ),
        boundary_conditions=[],
        ast_root=FieldMeta(
            name="test_field",
            label="Test Field",
            layout=FieldLayout.CELL,
            role=FieldRole.STATE,
        ),
    )
    solver = SolverFactory.create(solver_choice, eq_meta)
    assert isinstance(solver, ILinearSolver)
