"""Face-aligned material jumps must transmit the series-resistance flux."""

import pytest
import torch
from tests.conftest import small_gridfoam_config
from tests.helpers.configs import refined_config

from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.core.equation import equation
from gridfoam.core.field import CellField
from gridfoam.core.grid.factory import create_grid
from gridfoam.fv import fvc, fvm
from gridfoam.fv.fvm.laplacian import (
    _interpolate_gamma,  # pyright: ignore[reportPrivateUsage]
)
from gridfoam.fv.kernels.face_geometry import face_geometry
from gridfoam.meta.config import SolverConfig, fvSchemesConfig
from gridfoam.meta.enums import DomainBoundaryPatch, FieldRole, SolverType
from gridfoam.solvers.factory import create_solver


@pytest.mark.parametrize("contrast", [0.01, 100.0, 1e6])
@pytest.mark.parametrize("refined", [False, True])
@pytest.mark.parametrize("correction", ["corrected", "uncorrected"])
def test_two_material_diffusion_matches_exact_solution(
    contrast: float, refined: bool, correction: str
):
    scheme = f"Gauss harmonic {correction}"
    schemes = fvSchemesConfig.model_validate(
        {"laplacianSchemes": {"default": "linear", "laplacian(T)": scheme}}
    )
    config = (
        refined_config(
            fv_schemes=schemes,
            root_resolution=(4, 1, 1),
            domain_upper=(1.0, 0.25, 0.25),
            refinement_min=(0.5, 0.0, 0.0),
            refinement_max=(1.0, 0.25, 0.25),
        )
        if refined
        else small_gridfoam_config()
    )
    config = config.model_copy(
        update={
            "simulator": config.simulator.model_copy(
                update={"fvSchemes": schemes}
            )
        }
    )
    grid = create_grid(config)
    field = CellField(grid, "T", FieldRole.LOCAL, 1)
    field.add_boundary_conditions(
        {
            DomainBoundaryPatch.X_MINUS: DirichletBC(torch.tensor([0.0])),
            DomainBoundaryPatch.X_PLUS: DirichletBC(torch.tensor([1.0])),
        }
    )
    x = grid.cell_centers[:, :1]
    gamma = torch.where(
        x < 0.5, torch.ones_like(x), torch.full_like(x, contrast)
    )
    flux_density = 1.0 / (0.5 + 0.5 / contrast)
    exact = flux_density * torch.where(x < 0.5, x, 0.5 + (x - 0.5) / contrast)
    field.data = exact
    mat = fvm.laplacian(gamma, field)
    geo = face_geometry(grid)
    expected_flux = flux_density * geo.Sf_s[:, :1]
    torch.testing.assert_close(
        mat.flux(exact), expected_flux, atol=2e-9, rtol=1e-8
    )
    torch.testing.assert_close(
        mat.multiply(exact), mat.source, atol=2e-9, rtol=1e-8
    )
    if refined:
        interface = (x[geo.owner_s] < 0.5) != (x[geo.neighbour_s] < 0.5)
        assert interface.any()
        assert (geo.w_s[interface] - 0.5).abs().max() > 0.1

    # Solve from zero, rather than only applying the stencil to the exact field.
    field.data = torch.zeros_like(exact)
    solver = create_solver(
        SolverConfig(method=SolverType.CG, tolerance=1e-12, rel_tolerance=0.0)
    )
    result = solver.solve(equation(field, -fvm.laplacian(gamma, field)))
    torch.testing.assert_close(result.solution, exact, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("value", [0.0, 2.0, 1e200, 1e-200])
def test_harmonic_constant_coefficient_is_finite(value: float):
    grid = create_grid(small_gridfoam_config())
    gamma = torch.full_like(grid.cell_volumes, value, requires_grad=True)
    interpolated = _interpolate_gamma(face_geometry(grid), gamma, harmonic=True)
    torch.testing.assert_close(
        interpolated, torch.full_like(interpolated, value)
    )
    (gradient,) = torch.autograd.grad(interpolated.sum(), gamma)
    assert torch.isfinite(gradient).all()


def test_harmonic_insulating_interface_and_positive_coefficient_gradients():
    grid = create_grid(refined_config())
    geo = face_geometry(grid)
    gamma = torch.where(
        grid.cell_centers[:, :1] < 0.5, 0.0, 2.0
    ).requires_grad_()
    values = _interpolate_gamma(geo, gamma, harmonic=True)
    blocked = (gamma[geo.owner] == 0) | (gamma[geo.neighbour] == 0)
    assert torch.all(values[blocked] == 0)
    (gradient,) = torch.autograd.grad(values.sum(), gamma)
    assert torch.isfinite(gradient).all()

    positive = torch.linspace(
        1.0, 3.0, grid.num_cells, dtype=grid.dtype
    ).reshape(-1, 1)
    positive.requires_grad_()

    def interpolate(g: torch.Tensor) -> torch.Tensor:
        return _interpolate_gamma(geo, g, harmonic=True)

    assert torch.autograd.gradcheck(interpolate, (positive,))


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("tensor", [False, True])
def test_harmonic_rejects_invalid_diffusivity(value: float, tensor: bool):
    grid = create_grid(small_gridfoam_config())
    gamma = torch.full_like(grid.cell_volumes, value) if tensor else value
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _interpolate_gamma(face_geometry(grid), gamma, harmonic=True)


def test_corrected_harmonic_flux_uses_same_coefficient_for_hanging_correction():
    schemes = fvSchemesConfig.model_validate(
        {"laplacianSchemes": {"default": "Gauss harmonic corrected"}}
    )
    grid = create_grid(refined_config(fv_schemes=schemes))
    geo = face_geometry(grid)
    field = CellField(grid, "psi", FieldRole.LOCAL, 1)
    x = grid.cell_centers
    field.data = x[:, :1] ** 2 + x[:, 1:2]
    gamma = 1.0 + x[:, :1]
    gamma_f = _interpolate_gamma(geo, gamma, harmonic=True)[geo.single_idx]
    mat = fvm.laplacian(gamma, field)
    assert mat.face_flux_correction is not None
    torch.testing.assert_close(
        mat.flux(field.data),
        gamma_f * geo.mag_Sf_s * fvc.sn_grad(field).single_data,
        atol=1e-12,
        rtol=1e-12,
    )
    # No imposed boundary flux: internal fluxes cancel in the global balance.
    assert abs(float((mat.multiply(field.data) - mat.source).sum())) < 1e-12


@pytest.mark.parametrize("correction", ["corrected", "uncorrected"])
def test_gauss_linear_alias_preserves_existing_laplacian(correction: str):
    matrices = []
    for scheme in (correction, f"Gauss linear {correction}"):
        grid = create_grid(
            refined_config(
                fv_schemes=fvSchemesConfig.model_validate(
                    {"laplacianSchemes": {"default": scheme}}
                )
            )
        )
        field = CellField(grid, "psi", FieldRole.LOCAL, 1)
        field.data = grid.cell_centers[:, :1] ** 2
        matrices.append(fvm.laplacian(1.0 + field.data, field))
    for attr in ("diag", "upper", "lower", "source"):
        torch.testing.assert_close(
            getattr(matrices[0], attr), getattr(matrices[1], attr)
        )
