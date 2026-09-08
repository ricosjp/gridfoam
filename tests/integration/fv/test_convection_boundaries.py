"""Convection uses the boundary condition's face value in both directions."""

from pathlib import Path

import pytest
import torch
from tests.conftest import small_gridfoam_config
from tests.helpers.grids import immersed_plane_grid

from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.boundaries.basic.neumann import NeumannBC
from gridfoam.boundaries.derived.inlet_outlet import InletOutletBC
from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.factory import create_grid
from gridfoam.fv import fvm
from gridfoam.meta.enums import DivScheme, DomainBoundaryPatch, FieldRole


@pytest.mark.parametrize("side", ["domain", "upper", "lower"])
@pytest.mark.parametrize(
    "condition", ["fixed", "zero_gradient", "gradient", "inlet_outlet"]
)
@pytest.mark.parametrize("shape", [(), (3,), (3, 3)])
def test_boundary_transport_matches_prescribed_value(
    tmp_path: Path, side: str, condition: str, shape: tuple[int, ...]
):
    grid = immersed_plane_grid(tmp_path, 8, 0.3)
    field = CellField(grid, "transported", FieldRole.LOCAL, shape)
    cell_value = torch.full(shape, 3.0, dtype=grid.dtype)
    boundary_value = torch.full(shape, 7.0, dtype=grid.dtype)
    gradient = torch.full(shape, 2.0, dtype=grid.dtype)
    field.data = cell_value.expand_as(field.data).clone()
    phi = FaceField(grid, "phi", FieldRole.LOCAL, ())
    if side == "domain":
        patch = DomainBoundaryPatch.X_PLUS
        mask = grid.get_domain_bnd_mask(patch)
        cells = grid.domain_bnd_owner[mask]
        distance = torch.linalg.vector_norm(
            grid.domain_bnd_face_centers[mask] - grid.cell_centers[cells], dim=1
        )
        block = phi.domain_bnd_data
    else:
        patch = next(iter(grid.patch_name_to_id))
        mask = torch.ones(grid.num_immersed_faces, dtype=torch.bool)
        if side == "upper":
            cells = grid.owner[grid.ap_is_immersed_faces]
            distance = grid.ap_dist_owner_to_bnd
            block = phi.immersed_upper
        else:
            cells = grid.neighbour[grid.ap_is_immersed_faces]
            distance = grid.ap_dist_neighbour_to_bnd
            block = phi.immersed_lower
    if condition == "fixed":
        bc = DirichletBC(boundary_value)
    elif condition == "zero_gradient":
        bc = NeumannBC(torch.zeros_like(gradient))
    elif condition == "gradient":
        bc = NeumannBC(gradient)
    else:
        bc = InletOutletBC(boundary_value)
    field.add_boundary_conditions({patch: bc})

    # Change direction on the same field to also exercise inletOutlet's
    # dependency on the current flux, rather than a cached previous value.
    for flux in (2.0, -2.0, 0.0):
        block[mask] = flux
        matrix = fvm.div(phi, field)
        if condition == "fixed" or (condition == "inlet_outlet" and flux < 0):
            face_value = boundary_value.expand(cells.numel(), *shape)
        elif condition == "gradient":
            face_value = (
                cell_value
                + distance.reshape(-1, *(1 for _ in shape)) * gradient
            )
        else:
            face_value = cell_value.expand(cells.numel(), *shape)
        expected = torch.zeros_like(field.data)
        expected.index_add_(0, cells, flux * face_value)
        torch.testing.assert_close(
            matrix.multiply(field.data) - matrix.source,
            expected,
            atol=1e-12,
            rtol=0,
        )
        # A true fixed-value face must not introduce dependence on its cell.
        if condition == "fixed":
            torch.testing.assert_close(
                matrix.diag, torch.zeros_like(matrix.diag)
            )


def test_linear_advection_respects_fixed_outlet_value():
    config = small_gridfoam_config()
    config = config.model_copy(
        update={
            "simulator": config.simulator.model_copy(
                update={
                    "fvSchemes": config.simulator.fvSchemes.model_copy(
                        update={
                            "divSchemes": {"default": DivScheme.LINEAR},
                        }
                    ),
                }
            )
        }
    )
    grid = create_grid(config)
    field = CellField(grid, "q", FieldRole.LOCAL, ())
    field.data = grid.cell_centers[:, 0].clone()
    field.add_boundary_conditions(
        {
            DomainBoundaryPatch.X_MINUS: DirichletBC(torch.tensor(0.0)),
            DomainBoundaryPatch.X_PLUS: DirichletBC(torch.tensor(1.0)),
        }
    )
    phi = FaceField(grid, "phi", FieldRole.LOCAL, ())
    phi.single_data = grid.Sf[phi.single_mask, 0].clone()
    phi.domain_bnd_data = grid.domain_bnd_Sf[:, 0].clone()
    matrix = fvm.div(phi, field)
    derivative = (
        matrix.multiply(field.data) - matrix.source
    ) / grid.cell_volumes
    torch.testing.assert_close(
        derivative, torch.ones_like(derivative), atol=1e-12, rtol=0
    )
