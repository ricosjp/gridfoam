"""Integration tests for static vs dynamic IBM mesh construction."""

from __future__ import annotations

from pathlib import Path

import pytest
import pyvista as pv
import torch
from fluxel import Axis, quaternion_from_axis_angle
from tests.conftest import small_gridfoam_config

from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.factory import create_grid
from gridfoam.meta.config import DomainConfig, GridfoamConfig
from gridfoam.meta.enums import FieldRole, IbmType, MeshMotion


def _with_motion(config: GridfoamConfig, motion: MeshMotion) -> GridfoamConfig:
    return config.model_copy(
        update={
            "fluxel": config.fluxel.model_copy(update={"motion": motion}),
        }
    )


def test_static_grid_uses_one_shot_mesh_and_nested_ap_payload() -> None:
    grid = create_grid(small_gridfoam_config())
    assert isinstance(grid, AxisProjectedGrid)
    assert grid.is_dynamic is False
    assert grid.ib_translation == pytest.approx([0.0, 0.0, 0.0])
    assert grid.ib_rotation_quaternion == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert grid.ap_is_immersed_faces.shape[0] == grid.num_internal_faces
    assert grid.ap_dist_owner_to_bnd.shape[0] == grid.num_immersed_faces


def test_static_grid_rejects_ibm_updates() -> None:
    grid = create_grid(small_gridfoam_config())
    assert isinstance(grid, AxisProjectedGrid)
    with pytest.raises(RuntimeError, match="fluxel.motion=dynamic"):
        grid.update_ib(translation=[0.05, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="fluxel.motion=dynamic"):
        grid.remesh()


def test_update_ib_keeps_topology_on_empty_ibm() -> None:
    config = _with_motion(small_gridfoam_config(), MeshMotion.DYNAMIC)
    grid = create_grid(config)
    assert isinstance(grid, AxisProjectedGrid)
    assert grid.is_dynamic is True
    n_cells = grid.num_cells
    n_faces = grid.num_internal_faces

    grid.update_ib(translation=[0.05, 0.0, 0.0], warn_outside_refinement=False)

    assert grid.num_cells == n_cells
    assert grid.num_internal_faces == n_faces
    assert grid.ib_translation == pytest.approx([0.05, 0.0, 0.0])


def test_update_ib_accepts_axis_angle() -> None:
    config = _with_motion(small_gridfoam_config(), MeshMotion.DYNAMIC)
    grid = create_grid(config)
    assert isinstance(grid, AxisProjectedGrid)

    grid.update_ib(
        rotation_axis=Axis.Y,
        rotation_angle=45.0,
        degrees=True,
        warn_outside_refinement=False,
    )

    expected = quaternion_from_axis_angle(Axis.Y, 45.0, degrees=True)
    assert grid.ib_rotation_quaternion == pytest.approx(expected)


def test_update_ib_accepts_keyword_quaternion() -> None:
    config = _with_motion(small_gridfoam_config(), MeshMotion.DYNAMIC)
    grid = create_grid(config)
    assert isinstance(grid, AxisProjectedGrid)
    q = quaternion_from_axis_angle(Axis.Y, 45.0, degrees=True)

    grid.update_ib(rotation_quaternion=q, warn_outside_refinement=False)

    assert grid.ib_rotation_quaternion == pytest.approx(q)


def test_update_ib_rejects_mixed_rotation_args() -> None:
    config = _with_motion(small_gridfoam_config(), MeshMotion.DYNAMIC)
    grid = create_grid(config)
    assert isinstance(grid, AxisProjectedGrid)
    q = quaternion_from_axis_angle(Axis.Z, 90.0, degrees=True)

    with pytest.raises(ValueError, match="not both"):
        grid.update_ib(
            rotation_quaternion=q,
            rotation_axis=Axis.Z,
            rotation_angle=90.0,
            degrees=True,
            warn_outside_refinement=False,
        )
    with pytest.raises(ValueError, match="together"):
        grid.update_ib(rotation_axis=Axis.Z, warn_outside_refinement=False)


def test_update_ib_resizes_facefield_immersed_buffers(tmp_path: Path) -> None:
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
    assert grid.num_immersed_faces > 0

    n_cells = grid.num_cells
    n_faces = grid.num_internal_faces
    cell = CellField(grid, "T", FieldRole.LOCAL, 1)
    face = FaceField(grid, "phi", FieldRole.LOCAL, 1)
    cell.data[:] = 1.0
    preserved = face.single_data.clone()
    preserved[:] = 2.0
    face.single_data = preserved

    grid.update_ib(translation=[0.05, 0.0, 0.0], warn_outside_refinement=False)

    assert grid.num_cells == n_cells
    assert grid.num_internal_faces == n_faces
    assert cell.data.shape[0] == n_cells
    assert torch.allclose(cell.data, torch.ones_like(cell.data))
    assert face.immersed_upper.shape[0] == grid.num_immersed_faces
    assert face.immersed_lower.shape[0] == grid.num_immersed_faces
    assert face.single_mask.shape[0] == n_faces
    assert int(face.single_mask.sum().item()) == face.num_single_sided
    assert face.single_data.shape[0] == face.num_single_sided


def test_remesh_rebuilds_topology_for_uniform_mesh() -> None:
    config = _with_motion(small_gridfoam_config(), MeshMotion.DYNAMIC)
    grid = create_grid(config)
    assert isinstance(grid, AxisProjectedGrid)
    cell = CellField(grid, "p", FieldRole.LOCAL, 1)
    cell.data[:] = 3.0

    grid.remesh(
        rotation_axis=Axis.Z,
        rotation_angle=90.0,
        degrees=True,
        warn_outside_refinement=False,
    )

    expected = quaternion_from_axis_angle(Axis.Z, 90.0, degrees=True)
    assert cell.data.shape[0] == grid.num_cells
    assert torch.count_nonzero(cell.data) == 0
    assert grid.ib_rotation_quaternion == pytest.approx(expected)


def test_update_ib_moves_saved_surface_mesh(tmp_path: Path) -> None:
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

    rest_xmin = float(grid.surface_mesh.pvmesh.bounds[0])
    dx = 0.05
    grid.update_ib(translation=[dx, 0.0, 0.0], warn_outside_refinement=False)

    assert grid.surface_mesh.pvmesh.bounds[0] == pytest.approx(rest_xmin + dx)
    out = tmp_path / "surface.vtu"
    grid.surface_mesh.save(out, overwrite_features=True, overwrite_file=True)
    saved = pv.read(str(out))
    assert saved.bounds[0] == pytest.approx(rest_xmin + dx)
