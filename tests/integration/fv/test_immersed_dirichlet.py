"""Gibou boundary snapping, elliptic accuracy, and conservative projection."""

from pathlib import Path

import pytest
import torch
from tests.helpers.grids import immersed_plane_grid

from gridfoam.algorithms.utils.pressure_correction import (
    correct_phi,
    correct_velocity,
    solve_pressure_poisson,
)
from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.boundaries.basic.neumann import NeumannBC
from gridfoam.boundaries.derived.inlet_outlet import InletOutletBC
from gridfoam.core.equation import equation
from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.fv import fvc, fvm
from gridfoam.fv.boundary_ops import immersed_dirichlet_constraints
from gridfoam.meta.config import SolverConfig
from gridfoam.meta.enums import DomainBoundaryPatch as Patch
from gridfoam.meta.enums import FieldRole, SolverType
from gridfoam.solvers.base import LinearSolver
from gridfoam.solvers.factory import create_solver


def _scalar(
    grid: AxisProjectedGrid, name: str, interface_value: torch.Tensor
) -> CellField:
    q = CellField(grid, name, FieldRole.LOCAL, ())
    q.add_boundary_conditions(
        dict.fromkeys(Patch, NeumannBC(torch.tensor(0.0)))
    )
    q.add_boundary_conditions(
        {
            Patch.X_MINUS: DirichletBC(torch.tensor(0.0)),
            Patch.X_PLUS: DirichletBC(torch.tensor(1.0)),
        }
    )
    q.add_boundary_conditions(
        dict.fromkeys(grid.patch_name_to_id, DirichletBC(interface_value))
    )
    return q


def _solver() -> LinearSolver:
    return create_solver(
        SolverConfig(
            method=SolverType.CG,
            tolerance=1e-13,
            rel_tolerance=0.0,
            max_iter=1000,
        )
    )


def test_fixed_constraint_cache_refreshes_values_and_autograd(tmp_path: Path):
    grid = immersed_plane_grid(tmp_path, 8, theta=0.03)
    value = torch.tensor(0.7, dtype=grid.dtype, requires_grad=True)
    q = _scalar(grid, "q", value)
    patch = next(iter(grid.patch_name_to_id))
    bc = q.bcs[patch]
    assert isinstance(bc, DirichletBC)

    # Reuse the same field and BC, but start a fresh autograd graph each time.
    for factor in (2.0, 3.0):
        bc.value = factor * value
        cells, values = immersed_dirichlet_constraints(q)
        assert cells.numel() == 1
        torch.testing.assert_close(values, (factor * value).reshape(1))
        gradient = torch.autograd.grad(values.sum(), value)[0]
        torch.testing.assert_close(gradient, torch.full_like(value, factor))

    q.add_boundary_conditions({patch: NeumannBC(torch.tensor(0.0))})
    assert immersed_dirichlet_constraints(q)[0].numel() == 0
    q.add_boundary_conditions({patch: bc})
    assert immersed_dirichlet_constraints(q)[0].numel() == 1

    grid.ap_owner_near_boundary.zero_()
    grid.ap_neighbour_near_boundary.zero_()
    grid.invalidate_derived_caches()
    assert immersed_dirichlet_constraints(q)[0].numel() == 0


def test_mixed_constraint_selection_tracks_flow_reversal(tmp_path: Path):
    grid = immersed_plane_grid(tmp_path, 8, theta=0.03)
    q = CellField(grid, "q", FieldRole.LOCAL, ())
    phi = FaceField(grid, "phi", FieldRole.LOCAL, ())
    q.add_boundary_conditions(
        dict.fromkeys(
            grid.patch_name_to_id,
            InletOutletBC(torch.tensor(0.7, dtype=grid.dtype)),
        )
    )
    for flux, expected_cells in ((1.0, 0), (-1.0, 1), (1.0, 0)):
        phi.immersed_upper.fill_(flux)
        phi.immersed_lower.fill_(flux)
        cells, values = immersed_dirichlet_constraints(q)
        assert cells.numel() == expected_cells
        torch.testing.assert_close(values, torch.full_like(values, 0.7))


def test_near_boundary_poisson_is_second_order(tmp_path: Path):
    errors = []
    for n in (8, 16, 32):
        h = 1.0 / n
        grid = immersed_plane_grid(tmp_path / str(n), n, theta=0.4 * h)
        x_i = 0.5 - h / 2
        x_b = x_i + 0.4 * h * h
        q = _scalar(grid, "q", torch.tensor(x_b**2, dtype=grid.dtype))
        matrix = -fvm.laplacian(1.0, q)
        matrix.source = matrix.source - 2.0 * grid.cell_volumes
        eq = equation(q, matrix)
        torch.testing.assert_close(eq.fv_matrix.upper, eq.fv_matrix.lower)
        result = _solver().solve(eq)
        assert result.stats[0].converged
        cells, values = immersed_dirichlet_constraints(q)
        assert cells.numel() == 1
        torch.testing.assert_close(
            result.solution[cells], values, atol=1e-12, rtol=0
        )
        errors.append(
            float((result.solution - grid.cell_centers[:, 0] ** 2).abs().max())
        )
    assert errors[1] < 0.3 * errors[0]
    assert errors[2] < 0.3 * errors[1]


@pytest.mark.parametrize("theta", [0.3, 0.03, 0.0])
def test_immersed_pressure_projection_conserves_every_cell(
    tmp_path: Path, theta: float
):
    grid = immersed_plane_grid(tmp_path, 8, theta)
    p = _scalar(grid, "p", torch.tensor(0.7, dtype=grid.dtype))
    r = CellField(grid, "rAtU", FieldRole.LOCAL, ())
    r.data = torch.linspace(0.5, 1.5, grid.num_cells, dtype=grid.dtype)
    predicted = FaceField(grid, "predicted", FieldRole.LOCAL, ())
    predicted.single_data = torch.randn(
        predicted.single_data.shape,
        dtype=grid.dtype,
        generator=torch.Generator().manual_seed(7),
    )
    corrected = FaceField(grid, "corrected", FieldRole.LOCAL, ())
    result = solve_pressure_poisson(
        p,
        r,
        fvc.div(predicted).data,
        _solver(),
        n_non_orthogonal_correctors=0,
        p_needs_ref=False,
    )
    assert result.stats[0].converged
    correct_phi(corrected, predicted, result.matrix, p, r)
    torch.testing.assert_close(
        fvc.div(corrected).data, torch.zeros_like(p.data), atol=1e-8, rtol=0
    )
    cells, values = immersed_dirichlet_constraints(p)
    assert cells.numel() == (0 if theta == 0.3 else 1)
    torch.testing.assert_close(p.data[cells], values, atol=1e-12, rtol=0)
    assert torch.isfinite(fvc.grad(p).data).all()
    assert torch.isfinite(fvc.sn_grad(p).pack()).all()


def test_snapping_masks_and_distances_are_unit_invariant(tmp_path: Path):
    grids = [
        immersed_plane_grid(tmp_path / str(scale), 8, 0.03, scale)
        for scale in (1.0, 1000.0)
    ]
    a, b = grids
    torch.testing.assert_close(
        a.ap_owner_near_boundary, b.ap_owner_near_boundary
    )
    torch.testing.assert_close(
        a.ap_neighbour_near_boundary, b.ap_neighbour_near_boundary
    )
    torch.testing.assert_close(
        a.ap_dist_owner_to_bnd,
        b.ap_dist_owner_to_bnd / 1000.0,
        atol=1e-7,
        rtol=0,
    )
    assert float(a.ap_dist_owner_to_bnd[0]) < 0.01


def test_neumann_at_coincident_boundary_has_no_cell_constraint(tmp_path: Path):
    grid = immersed_plane_grid(tmp_path, 8, 0.0)
    q = _scalar(grid, "q", torch.tensor(0.0))
    q.add_boundary_conditions(
        dict.fromkeys(grid.patch_name_to_id, NeumannBC(torch.tensor(2.0)))
    )
    assert immersed_dirichlet_constraints(q)[0].numel() == 0
    matrix = fvm.laplacian(1.0, q)
    assert torch.isfinite(matrix.source).all()
    torch.testing.assert_close(
        fvc.sn_grad(q).immersed_upper,
        torch.full((grid.num_immersed_faces,), 2.0, dtype=grid.dtype),
    )


def test_prescribed_value_adjoint_matches_finite_difference(tmp_path: Path):
    grid = immersed_plane_grid(tmp_path, 8, 0.03)

    def objective(value: torch.Tensor) -> torch.Tensor:
        q = _scalar(grid, "adjoint_q", value)
        result = _solver().solve(equation(q, -fvm.laplacian(1.0, q)))
        return result.solution.square().sum()

    value = torch.tensor(0.7, dtype=grid.dtype, requires_grad=True)
    gradient = torch.autograd.grad(objective(value), value)[0]
    step = 1e-5
    finite_difference = (
        objective(value.detach() + step) - objective(value.detach() - step)
    ) / (2 * step)
    torch.testing.assert_close(gradient, finite_difference, atol=1e-8, rtol=0)


def test_velocity_correction_keeps_prescribed_cell_values(tmp_path: Path):

    grid = immersed_plane_grid(tmp_path, 8, 0.03)
    p = _scalar(grid, "p", torch.tensor(0.7))
    p.data = grid.cell_centers[:, 0].clone()
    U = CellField(grid, "U", FieldRole.LOCAL, (3,))
    value = torch.tensor([1.0, 2.0, 3.0], dtype=grid.dtype)
    U.add_boundary_conditions(
        dict.fromkeys(grid.patch_name_to_id, DirichletBC(value))
    )
    HbyA = CellField(grid, "HbyA", FieldRole.LOCAL, (3,))
    r = CellField(grid, "r", FieldRole.LOCAL, ())
    r.data = torch.ones_like(r.data)
    correct_velocity(U, HbyA, r, p)
    cells, values = immersed_dirichlet_constraints(U)
    torch.testing.assert_close(U.data[cells], values)
    assert cells.numel() == 1
