"""
Potential-flow initialization does not increase the initial flux divergence
norm.
"""

from __future__ import annotations

import pathlib

import torch
from tests.helpers import potential_flow_config

from gridfoam.core.field import get_or_create_facefield
from gridfoam.core.grid.factory import create_grid
from gridfoam.fv import fvc
from gridfoam.fv.flux import correct_flux
from gridfoam.meta.enums import FieldRole
from gridfoam.pre.potential_flow import PotentialFlow


def test_potential_flow_reduces_divergence(tmp_path: pathlib.Path) -> None:
    """
    Solving the elliptic potential-flow problem must not increase the L2
    norm of div(phi) compared to the initial flux from U.
    """
    grid = create_grid(potential_flow_config(tmp_path))
    solver = PotentialFlow(grid)

    phi = get_or_create_facefield(grid, "phi", FieldRole.LOCAL, ())
    correct_flux(phi, solver.U, update_internal=True)
    initial = torch.linalg.vector_norm(fvc.div(phi).data, ord=2).item()

    solver.solve()
    final = torch.linalg.vector_norm(fvc.div(solver.phi).data, ord=2).item()
    assert final <= initial + 1e-6
