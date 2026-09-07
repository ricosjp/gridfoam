"""Gibou boundary snapping, elliptic accuracy, and conservative projection."""

from pathlib import Path

import pytest
import pyvista as pv
import torch
from tests.conftest import small_gridfoam_config

from gridfoam.algorithms.utils.pressure_correction import (
    correct_phi,
    solve_pressure_poisson,
)
from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.boundaries.basic.neumann import NeumannBC
from gridfoam.core.equation import equation
from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.factory import create_grid
from gridfoam.fv import fvc, fvm
from gridfoam.fv.boundary_ops import immersed_dirichlet_constraints
from gridfoam.meta.config import DomainConfig, SolverConfig
from gridfoam.meta.enums import DomainBoundaryPatch as Patch
from gridfoam.meta.enums import FieldRole, SolverType
from gridfoam.solvers.factory import create_solver


def _plane_grid(
    path: Path, n: int, theta: float, scale: float = 1.0
) -> AxisProjectedGrid:
    path.mkdir(parents=True, exist_ok=True)
    interface = (0.5 - 0.5 / n + theta / n) * scale
    mesh_path = path / "plane.stl"
    pv.Plane(
        center=(interface, 0.05 * scale, 0.05 * scale),
        direction=(1, 0, 0),
        i_size=scale,
        j_size=scale,
    ).triangulate().save(mesh_path)
    config = small_gridfoam_config(output_dir=path)
    config = config.model_copy(
        update={
            "fluxel": config.fluxel.model_copy(
                update={
                    "domain": DomainConfig(
                        lower=[0, 0, 0], upper=[scale, 0.1 * scale, 0.1 * scale]
                    ),
                    "root_resolution": [n, 1, 1],
                    "mesh_path": mesh_path,
                }
            )
        }
    )
    grid = create_grid(config)
    assert isinstance(grid, AxisProjectedGrid)
    assert grid.num_immersed_faces == 1
    return grid


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


def _solver():
    return create_solver(
        SolverConfig(
            method=SolverType.CG,
            tolerance=1e-13,
            rel_tolerance=0.0,
            max_iter=1000,
        )
    )


def test_near_boundary_poisson_is_second_order(tmp_path: Path):
    errors = []
    for n in (8, 16, 32):
        h = 1.0 / n
        grid = _plane_grid(tmp_path / str(n), n, theta=0.4 * h)
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
    grid = _plane_grid(tmp_path, 8, theta)
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
        _plane_grid(tmp_path / str(scale), 8, 0.03, scale)
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
    grid = _plane_grid(tmp_path, 8, 0.0)
    q = _scalar(grid, "q", torch.tensor(0.0))
    q.add_boundary_conditions(
        dict.fromkeys(grid.patch_name_to_id, NeumannBC(torch.tensor(2.0)))
    )
    assert immersed_dirichlet_constraints(q)[0].numel() == 0
    matrix = fvm.laplacian(1.0, q)
    assert torch.isfinite(matrix.source).all()
    torch.testing.assert_close(
        fvc.sn_grad(q).immersed_upper, torch.full((1,), 2.0, dtype=grid.dtype)
    )
