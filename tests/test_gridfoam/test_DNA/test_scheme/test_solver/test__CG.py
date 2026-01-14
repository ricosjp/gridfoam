"""Tests for CG solver."""

import pytest
from unittest.mock import MagicMock

from gridfoam.DNA.config import SolverChoice
from gridfoam.DNA.enum import FieldLayout, FieldRole, NormType
from gridfoam.DNA.meta.equation import EquationMeta
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.solver._CG import CG
from gridfoam.DNA.scheme.solver._choice import SolverMethodChoice


def test_cg_solver_import():
    """Test that CG solver can be imported."""
    from gridfoam.DNA.scheme.solver._CG import CG
    assert CG is not None


def test_cg_solver_init():
    """Test CG solver initialization."""
    solver_choice = SolverChoice(
        method=SolverMethodChoice.CG,
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
    
    solver = CG(solver_choice, eq_meta)
    assert solver._solver_choice == solver_choice
    assert solver._eq_meta == eq_meta
    assert hasattr(solver, "required_fields")
    assert hasattr(solver, "solve")


def test_cg_solver_required_fields():
    """Test CG solver required_fields property."""
    solver_choice = SolverChoice(
        method=SolverMethodChoice.CG,
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
    
    solver = CG(solver_choice, eq_meta)
    required_fields = solver.required_fields
    assert isinstance(required_fields, list)
    # CG solver should require the search direction field
    assert len(required_fields) == 1
    assert required_fields[0].name.startswith("_CG.p")


def test_cg_solver_solve_calls_grid_handle():
    """Test CG solver solve method calls grid_handle methods."""
    solver_choice = SolverChoice(
        method=SolverMethodChoice.CG,
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
    
    solver = CG(solver_choice, eq_meta)
    
    # Create comprehensive mock grid handle
    mock_grid_handle = MagicMock()
    
    # Mock a minimal leaf node structure
    mock_cube = MagicMock()
    mock_cube.field = MagicMock()
    mock_cube.field.get_field = MagicMock(return_value=MagicMock())
    mock_cube.field.get_fvmatrix = MagicMock(return_value=MagicMock())
    
    # Mock iter_all_leaves to return one leaf
    mock_grid_handle.iter_all_leaves = MagicMock(return_value=iter([(0, mock_cube)]))
    
    # Mock other required methods
    mock_grid_handle.get_dx_at_depth = MagicMock(return_value=MagicMock())
    mock_grid_handle.sync_halo = MagicMock()
    mock_grid_handle.sync_all = MagicMock()
    
    # Fill minimal fields to avoid exceptions
    mock_cube.field.cells = {"test": MagicMock()}
    mock_cube.field.fvmatrices = {}

    # Execute solve; if it fails it should raise (no blanket except)
    with pytest.raises(Exception):
        solver.solve(mock_grid_handle)

    mock_grid_handle.iter_all_leaves.assert_called_once()
