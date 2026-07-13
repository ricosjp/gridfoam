"""Shared helpers for inlet-boundary optimization examples."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from jaxtyping import Float

from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.core.field import CellField, FaceField
from gridfoam.meta.enums import DomainBoundaryPatch


def configure_run_logger(
    name: str = "gridfoam.examples.optimize",
) -> logging.Logger:
    """Attach a single stream handler for example scripts."""
    log = logging.getLogger(name)
    formatter = logging.Formatter("%(message)s")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    log.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)
    return log


def set_inlet_dirichlet(
    field: CellField,
    patch: DomainBoundaryPatch,
    value: Float[torch.Tensor, " k"],
) -> None:
    """
    Replace the Dirichlet boundary condition on ``patch``.

    Parameters
    ----------
    field : CellField
        Field whose boundary condition is updated.
    patch : DomainBoundaryPatch
        Domain boundary patch (typically ``X_MINUS`` for inlet).
    value : torch.Tensor
        Boundary value tensor with shape ``[k]``. May require gradients.
    """
    field.bcs[patch] = DirichletBC(value)


def patch_cell_mean(
    field: CellField,
    patch: DomainBoundaryPatch,
    *,
    component: int = 0,
) -> Float[torch.Tensor, ""]:
    """
    Return the mean cell value adjacent to a domain boundary patch.

    Parameters
    ----------
    field : CellField
        Cell-centered field to sample.
    patch : DomainBoundaryPatch
        Domain boundary patch to average over.
    component : int, optional
        Component index for vector fields. Default is 0.

    Returns
    -------
    torch.Tensor
        Scalar mean (0-dim when ``component`` is fixed).
    """
    grid = field.grid
    mask = grid.get_domain_bnd_mask(patch)
    owner_cells = grid.domain_bnd_owner[mask]
    return field.data[owner_cells, component].mean()


@dataclass(frozen=True)
class FaceFieldSnapshot:
    """Detached copy of the face-field arrays used by channel-flow examples."""

    single_data: Float[torch.Tensor, " F_single k"]
    domain_bnd_data: Float[torch.Tensor, " F_bnd k"]


def snapshot_facefield(phi: FaceField) -> FaceFieldSnapshot:
    """Capture internal and domain-boundary face values."""
    return FaceFieldSnapshot(
        single_data=phi.single_data.clone().detach(),
        domain_bnd_data=phi.domain_bnd_data.clone().detach(),
    )


def restore_facefield(phi: FaceField, snapshot: FaceFieldSnapshot) -> None:
    """Restore face values from a previous snapshot."""
    phi.single_data = snapshot.single_data.clone()
    phi.domain_bnd_data = snapshot.domain_bnd_data.clone()
