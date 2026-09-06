"""Temporal order, variable-step weights, startup, and flux-history coupling."""

import math

import pytest
import torch
from tests.conftest import small_gridfoam_config

from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.core.equation import equation
from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.factory import create_grid
from gridfoam.fv import fvc, fvm
from gridfoam.fv.kernels.face_geometry import face_geometry
from gridfoam.fv.schemes.ddt import ddt_coefficients
from gridfoam.meta.config import SolverConfig, fvSchemesConfig
from gridfoam.meta.enums import DomainBoundaryPatch, FieldRole, SolverType
from gridfoam.solvers.factory import create_solver


def _grid(scheme: str) -> AxisProjectedGrid:
    config = small_gridfoam_config()
    config = config.model_copy(
        update={
            "simulator": config.simulator.model_copy(
                update={
                    "fvSchemes": fvSchemesConfig.model_validate(
                        {"ddtSchemes": {"default": scheme}}
                    )
                }
            )
        }
    )
    grid = create_grid(config)
    assert isinstance(grid, AxisProjectedGrid)
    return grid


def _set_dt(grid: AxisProjectedGrid, dt: float) -> None:
    # Exercise unequal intervals without introducing an adaptive-time runner.
    grid.sim_config = grid.sim_config.model_copy(
        update={
            "control": grid.sim_config.control.model_copy(update={"deltaT": dt})
        }
    )


@pytest.mark.parametrize(
    "scheme,minimum_ratio", [("euler", 1.8), ("backward", 3.7)]
)
def test_decay_has_expected_temporal_order(scheme: str, minimum_ratio: float):
    errors = []
    for n in (20, 40, 80):
        grid = _grid(scheme)
        _set_dt(grid, 1.0 / n)
        field = CellField(grid, "q", FieldRole.TRANSIENT, 1)
        field.data.fill_(1.0)
        field.update_history(reset=True)
        for _ in range(n):
            # Spatially uniform finite-volume reaction equation: dq/dt = -q.
            mat = fvm.ddt(field)
            field.data = mat.source / (mat.diag + grid.cell_volumes)
            field.update_history()
        errors.append(abs(field.data[0, 0].item() - math.exp(-1.0)))
    assert errors[0] / errors[1] > minimum_ratio
    assert errors[1] / errors[2] > minimum_ratio


def test_backward_variable_steps_are_exact_for_quadratic_and_constant_fields():
    grid = _grid("backward")
    field = CellField(grid, "q", FieldRole.TRANSIENT, 1)
    field.data.fill_(0.0)
    field.update_history(reset=True)
    assert ddt_coefficients(field) == (1.0, 1.0, 0.0)
    _set_dt(grid, 0.2)
    field.data.fill_(0.2**2)
    field.update_history()
    _set_dt(grid, 0.3)
    field.data.fill_(0.5**2)
    mat = fvm.ddt(field)
    derivative = (mat.multiply(field.data) - mat.source) / grid.cell_volumes
    torch.testing.assert_close(derivative, torch.ones_like(derivative))
    assert ddt_coefficients(field) == pytest.approx((1.6, 2.5, 0.9))

    field.reset_data([3.0])
    assert field.older_data is None
    assert ddt_coefficients(field) == (1.0, 1.0, 0.0)
    field.update_history()
    mat = fvm.ddt(field)
    torch.testing.assert_close(mat.multiply(field.data), mat.source)


def test_backward_source_differentiates_both_old_levels():
    grid = _grid("backward")
    field = CellField(grid, "q", FieldRole.TRANSIENT, 1)
    older = torch.full_like(field.data, 1.0, requires_grad=True)
    old = torch.full_like(field.data, 2.0, requires_grad=True)
    field.data = older
    field.update_history(reset=True)
    field.data = old
    field.update_history()
    mat = fvm.ddt(field)
    g_old, g_older = torch.autograd.grad(mat.source.sum(), (old, older))
    torch.testing.assert_close(g_old, 2.0 * grid.cell_volumes / grid.dt)
    torch.testing.assert_close(g_older, -0.5 * grid.cell_volumes / grid.dt)


@pytest.mark.parametrize("scheme", ["euler", "backward"])
def test_ddt_corr_uses_consistent_variable_step_history(scheme: str):
    grid = _grid(scheme)
    geo = face_geometry(grid)
    U = CellField(grid, "U", FieldRole.TRANSIENT, 3)
    phi = FaceField(grid, "phi", FieldRole.LOCAL, 1)

    U.data.fill_(1.0)
    flux_older = geo.Sf_s.sum(dim=1, keepdim=True)
    delta_older = 0.5 * flux_older.abs()
    phi.single_data = flux_older + delta_older
    U.update_history(reset=True)
    phi.update_history(reset=True)
    # First step has only Euler history, even under backward configuration.
    startup = fvc.ddt_corr(U, phi)
    expected_startup = (
        (1 - delta_older.abs() / phi.old_single_data.abs())
        * delta_older
        / grid.dt
    )
    torch.testing.assert_close(startup, expected_startup)

    _set_dt(grid, 0.2)
    U.data.fill_(2.0)
    flux_old = 2.0 * flux_older
    delta_old = 0.1 * flux_old.abs()
    phi.single_data = flux_old + delta_old
    U.update_history()
    phi.update_history()
    _set_dt(grid, 0.3)
    b, c = (2.5, 0.9) if scheme == "backward" else (1.0, 0.0)
    coupling = 1 - delta_old.abs() / phi.old_single_data.abs()
    expected = coupling * (b * delta_old - c * delta_older) / grid.dt
    torch.testing.assert_close(fvc.ddt_corr(U, phi), expected)
    if scheme == "backward":
        phi.update_history(reset=True)
        with pytest.raises(ValueError, match="synchronized"):
            fvc.ddt_corr(U, phi)


def test_ddt_field_override_and_topology_history_restart():
    grid = _grid("backward")
    field = CellField(grid, "q", FieldRole.TRANSIENT, 1)
    field.update_history()
    assert ddt_coefficients(field) == (1.5, 2.0, 0.5)
    grid.sim_config = grid.sim_config.model_copy(
        update={
            "fvSchemes": fvSchemesConfig.model_validate(
                {"ddtSchemes": {"default": "backward", "ddt(q)": "euler"}}
            )
        }
    )
    assert ddt_coefficients(field) == (1.0, 1.0, 0.0)
    field.sync_to_grid_topology(topology_changed=True)
    assert field.older_data is None
    assert field.previous_dt is None


def test_backward_rejects_nontransient_field():
    grid = _grid("backward")
    field = CellField(grid, "q", FieldRole.LOCAL, 1)
    with pytest.raises(ValueError, match="TRANSIENT"):
        fvm.ddt(field)


def test_backward_diffusion_has_second_order_time_convergence():
    errors = []
    for n in (10, 20, 40):
        grid = _grid("backward")
        _set_dt(grid, 0.5 / n)
        field = CellField(grid, "T", FieldRole.TRANSIENT, 1)
        field.add_boundary_conditions(
            {
                patch: DirichletBC(torch.zeros(1, dtype=grid.dtype))
                for patch in (
                    DomainBoundaryPatch.X_MINUS,
                    DomainBoundaryPatch.X_PLUS,
                )
            }
        )
        mode = torch.sin(math.pi * grid.cell_centers[:, :1])
        field.data = mode.clone()
        field.update_history(reset=True)
        solver = create_solver(
            SolverConfig(
                method=SolverType.CG, tolerance=1e-13, rel_tolerance=0.0
            )
        )
        for _ in range(n):
            mat = fvm.ddt(field) - fvm.laplacian(0.2, field)
            field.data = solver.solve(equation(field, mat)).solution
            field.update_history()
        # Exact decay rate of this fixed spatial stencil isolates time
        # error from spatial truncation error (h = 1/4 on this grid).
        rate = 0.2 * 4 / 0.25**2 * math.sin(math.pi * 0.25 / 2) ** 2
        exact = mode * math.exp(-rate * 0.5)
        errors.append(float((field.data - exact).abs().max()))
    assert errors[0] / errors[1] > 3.6
    assert errors[1] / errors[2] > 3.6
