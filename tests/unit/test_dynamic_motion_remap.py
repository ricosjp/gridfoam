"""
Volume-field remapping preserves approximate linear profiles after
immersed-body motion.
"""

from __future__ import annotations

from pathlib import Path

import pyvista as pv
import torch
from tests.conftest import small_gridfoam_config

from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.factory import create_grid
from gridfoam.core.grid.remap import capture_volume_fields, map_volume_fields
from gridfoam.meta.config import DomainConfig
from gridfoam.meta.enums import FieldRole, IbmType, MeshMotion


def test_remesh_maps_u_p_from_old_cell_centers(tmp_path: Path) -> None:
    """
    Remapped velocity and pressure stay near the linear fields and face
    flux stays finite.
    """
    stl_path = tmp_path / "cube.stl"
    box = pv.Box(bounds=(0.3, 0.7, 0.3, 0.7, 0.0, 0.1), quads=False)
    box.triangulate().save(str(stl_path))

    config = small_gridfoam_config(output_dir=tmp_path)
    config = config.model_copy(
        update={
            "fluxel": config.fluxel.model_copy(
                update={
                    "domain": DomainConfig(
                        lower=[0.0, 0.0, 0.0],
                        upper=[1.0, 1.0, 0.1],
                    ),
                    "mesh_path": stl_path,
                    "ibm_type": IbmType.AXIS_PROJECTED,
                    "motion": MeshMotion.DYNAMIC,
                }
            )
        }
    )
    grid = create_grid(config)
    assert isinstance(grid, AxisProjectedGrid)

    u_field = CellField(grid, "U", FieldRole.TRANSIENT, (3,))
    p_field = CellField(grid, "p", FieldRole.LOCAL, ())
    phi = FaceField(grid, "phi", FieldRole.LOCAL, ())
    u_field.data[:] = 0.0
    u_field.data[:, 0] = grid.cell_centers[:, 0]
    p_field.data[:] = grid.cell_centers[:, 1]
    phi.single_data[:] = 1.0
    phi.domain_bnd_data[:] = 0.5

    snapshot = capture_volume_fields(grid)
    grid.remesh(
        translation=[0.05, 0.0, 0.0],
        warn_outside_refinement=False,
    )
    map_volume_fields(grid, snapshot)

    ux_err = (u_field.data[:, 0] - grid.cell_centers[:, 0]).abs().max()
    p_err = (p_field.data[:] - grid.cell_centers[:, 1]).abs().max()
    assert ux_err.item() < 0.3
    assert p_err.item() < 0.3
    assert phi.single_data.shape[0] == phi.num_single_sided
    assert torch.isfinite(phi.single_data).all()
    assert not torch.allclose(
        phi.single_data, torch.zeros_like(phi.single_data)
    )
