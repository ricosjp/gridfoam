"""Track bounds and global balance while a scalar pulse crosses refinement."""

import math

import pytest
import torch
from tests.conftest import small_gridfoam_config
from tests.helpers.configs import refined_config

from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.boundaries.basic.neumann import NeumannBC
from gridfoam.core.dimensions import DIM_VOL_FLUX
from gridfoam.core.equation import equation
from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.factory import create_grid
from gridfoam.fv import fvm
from gridfoam.meta.config import SolverConfig, fvSchemesConfig
from gridfoam.meta.enums import (
    DivScheme,
    DomainBoundaryPatch,
    FieldRole,
    SolverType,
)
from gridfoam.solvers.factory import create_solver

_ROOT_CELLS = 16
_SLAB = 0.1
# Occupies coarse cells [4, 7), leaving one empty coarse cell before x = 1/2.
_PULSE_FIRST_CELL = 4
_PULSE_END_CELL = 7
_INTERFACE_CELL = _ROOT_CELLS // 2
_COARSE_CFL = 0.1
# Trailing edge starts in cell 4; finish one coarse cell past the interface.
_N_STEPS = math.ceil((_INTERFACE_CELL + 1 - _PULSE_FIRST_CELL) / _COARSE_CFL)
_SOLVER_ATOL = 1e-13
_MASS_RTOL = 1e-8
_BOUNDS_ATOL = 1e-10
_LINEAR_OVERSHOOT = 0.01


def pulse_metrics(scheme: DivScheme, refined: bool) -> dict[str, float]:
    dx = 1.0 / _ROOT_CELLS
    pulse_left = _PULSE_FIRST_CELL * dx
    pulse_right = _PULSE_END_CELL * dx
    interface = _INTERFACE_CELL * dx
    dt = _COARSE_CFL * dx
    domain_upper = (1.0, _SLAB, _SLAB)
    config = (
        refined_config(
            root_resolution=(_ROOT_CELLS, 1, 1),
            domain_upper=domain_upper,
            refinement_min=(interface, 0.0, 0.0),
            refinement_max=(1.0, _SLAB, _SLAB),
        )
        if refined
        else small_gridfoam_config()
    )
    config = config.model_copy(
        update={
            "fluxel": config.fluxel.model_copy(
                update={
                    "root_resolution": [_ROOT_CELLS, 1, 1],
                    "domain": config.fluxel.domain.model_copy(
                        update={"upper": list(domain_upper)}
                    ),
                }
            ),
            "simulator": config.simulator.model_copy(
                update={
                    "fvSchemes": fvSchemesConfig(
                        divSchemes={"default": scheme}
                    ),
                    "control": config.simulator.control.model_copy(
                        update={"deltaT": dt}
                    ),
                }
            ),
        }
    )
    grid = create_grid(config)
    q = CellField(grid, "q", FieldRole.TRANSIENT, ())
    q.add_boundary_conditions(
        {
            patch: (
                DirichletBC(torch.zeros((), dtype=grid.dtype))
                if patch == DomainBoundaryPatch.X_MINUS
                else NeumannBC(torch.zeros((), dtype=grid.dtype))
            )
            for patch in DomainBoundaryPatch
        }
    )
    x = grid.cell_centers[:, 0]
    q.data = ((x >= pulse_left) & (x < pulse_right)).to(grid.dtype)
    q.update_history(reset=True)
    phi = FaceField(grid, "phi", FieldRole.LOCAL, (), dimension=DIM_VOL_FLUX)
    phi.single_data = grid.Sf[phi.single_mask, 0]
    phi.domain_bnd_data = grid.domain_bnd_Sf[:, 0]
    solver = create_solver(
        SolverConfig(
            method=SolverType.BiCGSTAB,
            tolerance=_SOLVER_ATOL,
            rel_tolerance=0.0,
        )
    )
    volume = grid.cell_volumes
    initial_mass = float((q.data * volume).sum())
    minimum, maximum = 0.0, 1.0
    mass_error, boundary_transport = 0.0, 0.0
    for _ in range(_N_STEPS):
        # The library's Euler + implicit upwind/deferred-TVD step, with one
        # deferred correction evaluated from the previous time level.
        mat = fvm.ddt(q) + fvm.div(phi, q)
        result = solver.solve(equation(q, mat))
        assert all(s.converged for s in result.stats)
        q.data = result.solution
        minimum = min(minimum, float(q.data.min()))
        maximum = max(maximum, float(q.data.max()))
        # Inflow is fixed to zero; outflow uses the solved cell value.
        boundary_transport += grid.dt * float(
            (
                torch.clamp(phi.domain_bnd_data, min=0.0)
                * q.data[grid.domain_bnd_owner]
            ).sum()
        )
        balance = (
            float((q.data * volume).sum()) + boundary_transport - initial_mass
        )
        mass_error = max(mass_error, abs(balance) / initial_mass)
        q.update_history()
    assert (
        float((q.data[x > interface] * volume[x > interface]).sum())
        > 0.5 * initial_mass
    )
    return {"min": minimum, "max": maximum, "relative_mass_error": mass_error}


@pytest.mark.parametrize("refined", [False, True])
@pytest.mark.parametrize(
    "scheme", [s for s in DivScheme if s != DivScheme.LINEAR]
)
def test_pulse_bounds_and_conservation(
    scheme: DivScheme, refined: bool
) -> None:
    """
    Upwind and TVD pulses stay bounded and conserve mass across refinement.
    """
    metrics = pulse_metrics(scheme, refined)
    assert metrics["relative_mass_error"] < _MASS_RTOL, metrics
    assert metrics["min"] >= -_BOUNDS_ATOL, metrics
    assert metrics["max"] <= 1 + _BOUNDS_ATOL, metrics


@pytest.mark.parametrize("refined", [False, True])
def test_linear_advection_exposes_unbounded_pulse(refined: bool) -> None:
    """
    Negative control: the diagnostic must detect central-scheme
    oscillations.
    """
    metrics = pulse_metrics(DivScheme.LINEAR, refined)
    assert metrics["relative_mass_error"] < _MASS_RTOL, metrics
    assert metrics["min"] < -_LINEAR_OVERSHOOT, metrics
    assert metrics["max"] > 1 + _LINEAR_OVERSHOOT, metrics
