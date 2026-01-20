"""Tests for BiCGSTAB solver."""



from gridfoam.DNA.config import SolverChoice
from gridfoam.DNA.enum import FieldLayout, FieldRole, NormType
from gridfoam.DNA.meta.equation import EquationMeta
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.solver._BiCGSTAB import BiCGSTAB
from gridfoam.DNA.scheme.solver._choice import SolverMethodChoice


def test_bicgstab_solver_init():
    """Test BiCGSTAB solver initialization."""
    solver_choice = SolverChoice(
        method=SolverMethodChoice.BICGSTAB,
        tolerance=1e-6,
        rel_tolerance=1e-6,
        max_iter=1000,
        norm_type=NormType.L_2,
    )
    target_field = FieldMeta(
        name="test",
        label="Test",
        layout=FieldLayout.CELL,
        role=FieldRole.STATE,
    )
    eq_meta = EquationMeta(
        name="test_eq",
        target_field=target_field,
        boundary_conditions=[],
        ast_root=target_field,
    )

    solver = BiCGSTAB(solver_choice, eq_meta)
    assert solver._eq_meta == eq_meta
    assert solver._tolerance == solver_choice.tolerance
    assert solver._rel_tolerance == solver_choice.rel_tolerance
    assert solver._max_iter == solver_choice.max_iter
    assert hasattr(solver, "required_fields")
    assert hasattr(solver, "solve")


def test_bicgstab_solver_required_fields():
    """Test BiCGSTAB solver required_fields property."""
    solver_choice = SolverChoice(
        method=SolverMethodChoice.BICGSTAB,
        tolerance=1e-6,
        rel_tolerance=1e-6,
        max_iter=1000,
        norm_type=NormType.L_2,
    )
    target_field = FieldMeta(
        name="test",
        label="Test",
        layout=FieldLayout.CELL,
        role=FieldRole.STATE,
    )
    eq_meta = EquationMeta(
        name="test_eq",
        target_field=target_field,
        boundary_conditions=[],
        ast_root=target_field,
    )

    solver = BiCGSTAB(solver_choice, eq_meta)
    required_fields = solver.required_fields
    assert isinstance(required_fields, list)
    # BiCGSTAB solver should require auxiliary fields
    assert len(required_fields) >= 1
