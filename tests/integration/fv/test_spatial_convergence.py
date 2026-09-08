"""Manufactured Poisson solution on Cartesian and octree grid sequences."""

import math

import pytest
import torch
from tests.conftest import small_gridfoam_config
from tests.helpers.configs import refined_config

from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.core.equation import equation
from gridfoam.core.field import CellField
from gridfoam.core.grid.factory import create_grid
from gridfoam.fv import fvm
from gridfoam.fv.kernels.face_geometry import face_geometry
from gridfoam.meta.config import SolverConfig
from gridfoam.meta.enums import DomainBoundaryPatch, FieldRole, SolverType
from gridfoam.solvers.factory import create_solver

_ROOT_RESOLUTIONS = (8, 16, 32)
_MAX_PICARD = 100
_EQUATION_ATOL = 1e-8
_SOLVER_ATOL = 1e-13
_MIN_OBSERVED_ORDER = 1.8


def poisson_errors(n: int, refined: bool) -> dict[str, float]:
    config = (
        refined_config(root_resolution=(n, n, 1))
        if refined
        else small_gridfoam_config()
    )
    if not refined:
        config = config.model_copy(
            update={
                "fluxel": config.fluxel.model_copy(
                    update={"root_resolution": [n, n, 1]}
                )
            }
        )
    grid = create_grid(config)
    field = CellField(grid, "q", FieldRole.LOCAL, ())
    field.add_boundary_conditions(
        {
            patch: DirichletBC(torch.zeros((), dtype=grid.dtype))
            for patch in (
                DomainBoundaryPatch.X_MINUS,
                DomainBoundaryPatch.X_PLUS,
                DomainBoundaryPatch.Y_MINUS,
                DomainBoundaryPatch.Y_PLUS,
            )
        }
    )
    x = grid.cell_centers
    exact = torch.sin(math.pi * x[:, 0]) * torch.sin(math.pi * x[:, 1])
    # Continuous manufactured forcing, integrated with midpoint quadrature.
    forcing = 2 * math.pi**2 * exact * grid.cell_volumes
    solver = create_solver(
        SolverConfig(
            method=SolverType.CG, tolerance=_SOLVER_ATOL, rel_tolerance=0.0
        )
    )
    for _iteration in range(_MAX_PICARD):
        mat = -fvm.laplacian(1.0, field)
        mat.source = mat.source + forcing
        field.data = solver.solve(equation(field, mat)).solution
        # Reassemble the deferred correction at the NEW solution before
        # judging convergence, not merely the frozen linear-system residual.
        check = -fvm.laplacian(1.0, field)
        residual = (
            check.multiply(field.data) - check.source - forcing
        ) / grid.cell_volumes
        if float(residual.abs().max()) < _EQUATION_ATOL:
            break
    else:
        raise AssertionError("Deferred diffusion correction did not converge")
    error = (field.data - exact).abs()
    volume = grid.cell_volumes
    geo = face_geometry(grid)
    # Analytic face-integrated flux: integrate the tangential sine exactly.
    centers = grid.face_centers[geo.single_idx]
    axis = grid.axis[geo.single_idx]
    sizes = torch.minimum(
        grid.cell_sizes[geo.owner_s], grid.cell_sizes[geo.neighbour_s]
    )
    flux = torch.zeros_like(geo.mag_Sf_s)
    for normal, tangent in ((0, 1), (1, 0)):
        mask = axis == normal
        flux[mask] = (
            math.pi
            * torch.cos(math.pi * centers[mask, normal])
            * torch.sin(math.pi * centers[mask, tangent])
            * torch.sinc(sizes[mask, tangent] / 2)
            * geo.Sf_s[mask, normal]
        )
    flux_error = (
        fvm.laplacian(1.0, field).flux(field.data) - flux
    ) / geo.mag_Sf_s
    result = {
        "h": 1 / n,
        "cells": float(grid.num_cells),
        "l1": float((error * volume).sum() / volume.sum()),
        "l2": float(torch.sqrt((error.square() * volume).sum() / volume.sum())),
        "linf": float(error.max()),
        "flux_l2": float(
            torch.sqrt(
                (flux_error.square() * geo.mag_Sf_s).sum() / geo.mag_Sf_s.sum()
            )
        ),
        "residual_linf": float(residual.abs().max()),
        "passes": float(_iteration + 1),
    }
    if geo.num_hanging:
        e = flux_error[geo.hang_idx]
        area = geo.mag_Sf_s[geo.hang_idx]
        result["hanging_flux_l2"] = float(
            torch.sqrt((e.square() * area).sum() / area.sum())
        )
    return result


@pytest.mark.parametrize("refined", [False, True])
def test_manufactured_poisson_solution_converges_quadratically(
    refined: bool,
) -> None:
    """
    Cell errors approach second order and face-flux errors decrease on both
    grid families.
    """
    results = [poisson_errors(n, refined) for n in _ROOT_RESOLUTIONS]
    for norm in ("l1", "l2", "linf"):
        for coarse, fine in zip(results[:-1], results[1:], strict=True):
            order = math.log2(coarse[norm] / fine[norm])
            assert order > _MIN_OBSERVED_ORDER, (refined, norm, order, results)

    # Flux errors must decrease, but cell-value order does not establish
    # second-order face gradients at coarse/fine interfaces.
    for norm in ["flux_l2", "hanging_flux_l2"] if refined else ["flux_l2"]:
        for coarse, fine in zip(results[:-1], results[1:], strict=True):
            assert fine[norm] < coarse[norm], (refined, norm, results)
