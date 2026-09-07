"""Unit tests for the per-field boundary-state cache in ``boundary_ops``."""

from __future__ import annotations

import pathlib

import torch
from tests.helpers import channel_config

from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.base import IGridBase
from gridfoam.core.grid.factory import create_grid
from gridfoam.fv import fvc
from gridfoam.fv.boundary_ops import iter_boundary_states
from gridfoam.meta.enums import DomainBoundaryPatch, FaceSide, FieldRole
from gridfoam.meta.types import PatchName


class _CountingDirichlet(DirichletBC):
    def __init__(self, value: torch.Tensor):
        super().__init__(value)
        self.calls = 0

    def evaluate(
        self,
        field: CellField,
        patch_name: PatchName,
        side: FaceSide = FaceSide.UPPER,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.calls += 1
        return super().evaluate(field, patch_name, side)


def _scalar_field(
    grid: IGridBase, name: str
) -> tuple[CellField, _CountingDirichlet]:
    field = CellField(grid, name, FieldRole.LOCAL, ())
    field.data = grid.cell_centers[:, 0].clone()
    bc = _CountingDirichlet(torch.tensor(2.0, dtype=grid.dtype))
    field.add_boundary_conditions({DomainBoundaryPatch.X_MINUS: bc})
    return field, bc


def test_boundary_state_is_evaluated_once_per_data_generation(
    tmp_path: pathlib.Path,
):
    # Several operators in one solver step share a single BC evaluation.
    grid = create_grid(channel_config(tmp_path))
    field, bc = _scalar_field(grid, "psi_cache")

    fvc.interpolate(field)
    fvc.grad(field)
    fvc.sn_grad(field)
    list(iter_boundary_states(field))
    assert bc.calls == 1


def test_boundary_state_cache_invalidates_on_data_change(
    tmp_path: pathlib.Path,
):
    grid = create_grid(channel_config(tmp_path))
    field, bc = _scalar_field(grid, "psi_cache_inval")
    fvc.interpolate(field)
    assert bc.calls == 1

    # Replacing the tensor invalidates the cache ...
    field.data = field.data * 2.0
    fvc.interpolate(field)
    assert bc.calls == 2

    # ... and so does an in-place modification.
    field.data[:] = 1.0
    states = list(iter_boundary_states(field))
    assert bc.calls == 3
    torch.testing.assert_close(
        states[0].psi_b, torch.full_like(states[0].psi_b, 2.0)
    )


def test_boundary_state_cache_tracks_bc_dependencies(tmp_path: pathlib.Path):
    # inletOutlet reads ``phi``; changing phi in place must re-evaluate the
    # velocity boundary state even though U itself is unchanged.
    grid = create_grid(channel_config(tmp_path))
    U = CellField(grid, "U", FieldRole.LOCAL, (3,))  # BCs from the config
    phi = FaceField(grid, "phi", FieldRole.LOCAL, ())
    outlet = grid.get_domain_bnd_mask(DomainBoundaryPatch.X_PLUS)
    U.data = torch.ones_like(U.data)

    phi.domain_bnd_data[outlet] = 1.0  # outflow -> zero gradient -> U_P
    outflow_state = [
        s
        for s in iter_boundary_states(U)
        if s.batch.patch_name == DomainBoundaryPatch.X_PLUS
    ][0]
    torch.testing.assert_close(
        outflow_state.psi_b, torch.ones_like(outflow_state.psi_b)
    )

    phi.domain_bnd_data[outlet] = -1.0  # inflow -> fixed inlet value (0)
    inflow_state = [
        s
        for s in iter_boundary_states(U)
        if s.batch.patch_name == DomainBoundaryPatch.X_PLUS
    ][0]
    torch.testing.assert_close(
        inflow_state.psi_b, torch.zeros_like(inflow_state.psi_b)
    )
