"""Shared helpers for moving-boundary IBM examples."""

from __future__ import annotations

import logging
from pathlib import Path

import pyvista as pv
import torch

from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase
from gridfoam.fv.flux import correct_flux
from gridfoam.io.vtu import save_export_fields_as_vtu, to_unstructured_grid


def configure_run_logger(
    name: str = "gridfoam.examples.dynamic_motions",
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


def write_square_prism_stl(
    path: Path,
    *,
    bounds: tuple[float, float, float, float, float, float],
) -> Path:
    """
    Write an axis-aligned square prism STL used as the immersed body.

    Parameters
    ----------
    path : pathlib.Path
        Destination STL path.
    bounds : tuple of float
        PyVista box bounds ``(xmin, xmax, ymin, ymax, zmin, zmax)``.

    Returns
    -------
    pathlib.Path
        The written STL path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    box = pv.Box(bounds=bounds, quads=False)
    box = box.triangulate()
    box.save(str(path))
    return path


def set_immersed_wall_velocity(
    grid: AxisProjectedGrid,
    velocity: list[float],
    patch_name: str = "_default",
) -> None:
    """Set the Dirichlet wall velocity on the immersed patch."""
    field = grid.get_cellfield("U")
    if field is None:
        return
    bc = field.bcs.get(patch_name)
    if isinstance(bc, DirichletBC):
        bc.value = torch.tensor(velocity, dtype=grid.dtype, device=grid.device)


def refresh_flux(grid: IGridBase, *, update_internal: bool = True) -> None:
    """Synchronize ``phi`` from ``U`` after an IBM or topology update."""
    u_field = grid.get_cellfield("U")
    phi = grid.get_facefield("phi")
    if u_field is None or phi is None:
        return
    correct_flux(phi, u_field, update_internal=update_internal)


def write_step_outputs(
    grid: AxisProjectedGrid,
    *,
    output_dir: Path,
    base_name: str,
    step: int,
) -> None:
    """Write volume VTU and the posed surface mesh for one step."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_export_fields_as_vtu(
        grid,
        str(output_dir / f"{base_name}_{step:04d}.vtu"),
        ugrid=to_unstructured_grid(grid),
    )
    grid.surface_mesh.save(
        output_dir / f"{base_name}_{step:04d}_surface.vtu",
        overwrite_features=True,
        overwrite_file=True,
    )
