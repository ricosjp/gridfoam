"""Unit tests for ``FaceField`` packed layout."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self

import graphlow as gl
import pytest
import pyvista as pv
import torch
from jaxtyping import Bool, Float, Int
from tests.conftest import small_gridfoam_config

from gridfoam.core.field import CellField, FaceField, packed_face_n_rows
from gridfoam.core.fv_cache import FvGridCache
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase
from gridfoam.core.grid.factory import create_grid
from gridfoam.meta.config import DomainConfig, SimulatorConfig
from gridfoam.meta.enums import DomainBoundaryPatch, FieldRole, IbmType


class _NonAxisProjectedGrid(IGridBase):
    """Minimal non-AP grid so pack omits immersed blocks."""

    def __init__(self) -> None:
        self._device = torch.device("cpu")
        self._dtype = torch.float64
        self._n_cells = 4
        self._n_internal = 4
        self._n_bnd = 8
        self._fv_cache = FvGridCache()

    def register_cellfield(self, field: CellField) -> None:
        return None

    def register_facefield(self, field: FaceField) -> None:
        return None

    def get_cellfield(self, name: str) -> CellField | None:
        return None

    def get_facefield(self, name: str) -> FaceField | None:
        return None

    def cellfield_names(self) -> Iterator[str]:
        return iter(())

    def facefield_names(self) -> Iterator[str]:
        return iter(())

    def get_domain_bnd_mask(
        self, patch_name: DomainBoundaryPatch
    ) -> Bool[torch.Tensor, " F"]:
        return torch.zeros(self._n_bnd, dtype=torch.bool, device=self._device)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> Self:
        return self

    @property
    def surface_mesh(self) -> gl.TensorMesh[Any]:
        raise NotImplementedError

    @property
    def sim_config(self) -> SimulatorConfig:
        raise NotImplementedError

    @property
    def dt(self) -> float:
        return 0.0

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def fv_cache(self) -> FvGridCache:
        return self._fv_cache

    @property
    def num_cells(self) -> int:
        return self._n_cells

    @property
    def num_internal_faces(self) -> int:
        return self._n_internal

    @property
    def num_domain_bnd_faces(self) -> int:
        return self._n_bnd

    @property
    def patch_name_to_id(self) -> dict[str, int]:
        return {}

    @property
    def owner(self) -> Int[torch.Tensor, " F"]:
        return torch.zeros(self._n_internal, dtype=torch.long)

    @property
    def neighbour(self) -> Int[torch.Tensor, " F"]:
        return torch.zeros(self._n_internal, dtype=torch.long)

    @property
    def axis(self) -> Int[torch.Tensor, " F"]:
        return torch.zeros(self._n_internal, dtype=torch.long)

    @property
    def domain_bnd_owner(self) -> Int[torch.Tensor, " F_bnd"]:
        return torch.zeros(self._n_bnd, dtype=torch.long)

    @property
    def domain_bnd_dir_id(self) -> Int[torch.Tensor, " F_bnd"]:
        return torch.zeros(self._n_bnd, dtype=torch.long)

    @property
    def cell_centers(self) -> Float[torch.Tensor, " C 3"]:
        return torch.zeros((self._n_cells, 3), dtype=self._dtype)

    @property
    def cell_sizes(self) -> Float[torch.Tensor, " C 3"]:
        return torch.zeros((self._n_cells, 3), dtype=self._dtype)

    @property
    def cell_volumes(self) -> Float[torch.Tensor, " C"]:
        return torch.zeros(self._n_cells, dtype=self._dtype)

    @property
    def face_centers(self) -> Float[torch.Tensor, " F 3"]:
        return torch.zeros((self._n_internal, 3), dtype=self._dtype)

    @property
    def Sf(self) -> Float[torch.Tensor, "F 3"]:
        return torch.zeros((self._n_internal, 3), dtype=self._dtype)

    @property
    def domain_bnd_face_centers(self) -> Float[torch.Tensor, " F_bnd 3"]:
        return torch.zeros((self._n_bnd, 3), dtype=self._dtype)

    @property
    def domain_bnd_Sf(self) -> Float[torch.Tensor, " F_bnd 3"]:
        return torch.zeros((self._n_bnd, 3), dtype=self._dtype)


def _axis_projected_grid_with_ib(tmp_path: Path) -> AxisProjectedGrid:
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
                }
            )
        }
    )
    grid = create_grid(config)
    assert isinstance(grid, AxisProjectedGrid)
    return grid


def test_pack_row_count_without_immersed_blocks() -> None:
    # Non-AP grids pack only single-sided and domain-boundary faces.
    grid = _NonAxisProjectedGrid()
    phi = FaceField(grid, "phi", FieldRole.LOCAL, 1)
    n_expected = phi.num_single_sided + grid.num_domain_bnd_faces
    assert phi.pack().shape[0] == n_expected
    assert packed_face_n_rows(grid) == n_expected
    assert not hasattr(phi, "_immersed_upper")


def test_pack_row_count_on_axis_projected_grid(tmp_path: Path) -> None:
    # AP layout adds both immersed sides: F_single + F_bnd + 2 * F_immersed.
    grid = _axis_projected_grid_with_ib(tmp_path)
    phi = FaceField(grid, "phi", FieldRole.LOCAL, 1)
    n_expected = (
        phi.num_single_sided
        + grid.num_domain_bnd_faces
        + 2 * grid.num_immersed_faces
    )
    assert grid.num_immersed_faces > 0
    assert phi.packed_n_rows() == n_expected
    assert phi.pack().shape[0] == n_expected
    assert packed_face_n_rows(grid) == n_expected


def test_unpack_roundtrip_preserves_blocks(tmp_path: Path) -> None:
    # unpack(pack(phi)) restores every face block.
    grid = _axis_projected_grid_with_ib(tmp_path)
    phi = FaceField(grid, "phi", FieldRole.LOCAL, 2)
    phi.single_data[:] = 1.0
    phi.domain_bnd_data[:] = 2.0
    phi.immersed_upper[:] = 3.0
    phi.immersed_lower[:] = 4.0
    single = phi.single_data.clone()
    bnd = phi.domain_bnd_data.clone()
    upper = phi.immersed_upper.clone()
    lower = phi.immersed_lower.clone()

    phi.unpack(phi.pack())

    assert torch.equal(phi.single_data, single)
    assert torch.equal(phi.domain_bnd_data, bnd)
    assert torch.equal(phi.immersed_upper, upper)
    assert torch.equal(phi.immersed_lower, lower)


def test_unpack_rejects_wrong_row_count(
    small_axis_projected_grid: AxisProjectedGrid,
) -> None:
    # Row count must match packed_n_rows.
    phi = FaceField(
        small_axis_projected_grid, "phi_pack_bad", FieldRole.LOCAL, 1
    )
    packed = phi.pack()
    with pytest.raises(ValueError, match="rows"):
        phi.unpack(packed[:-1])


def test_unpack_rejects_wrong_feature_count(
    small_axis_projected_grid: AxisProjectedGrid,
) -> None:
    # Feature axis k must match num_components.
    phi = FaceField(small_axis_projected_grid, "phi_pack_k", FieldRole.LOCAL, 1)
    packed = torch.zeros(
        (phi.packed_n_rows(), 2), dtype=phi.grid.dtype, device=phi.grid.device
    )
    with pytest.raises(ValueError, match="features"):
        phi.unpack(packed)


def test_pack_layout_is_single_then_bnd_then_immersed(
    tmp_path: Path,
) -> None:
    # Packed axis-0 order is [single | domain_bnd | upper | lower].
    grid = _axis_projected_grid_with_ib(tmp_path)
    phi = FaceField(grid, "phi", FieldRole.LOCAL, 1)
    phi.single_data[:] = 1.0
    phi.domain_bnd_data[:] = 2.0
    phi.immersed_upper[:] = 3.0
    phi.immersed_lower[:] = 4.0
    packed = phi.pack()
    n_single = phi.num_single_sided
    n_bnd = grid.num_domain_bnd_faces
    n_ib = grid.num_immersed_faces
    assert torch.equal(packed[:n_single], phi.single_data)
    assert torch.equal(packed[n_single : n_single + n_bnd], phi.domain_bnd_data)
    assert torch.equal(
        packed[n_single + n_bnd : n_single + n_bnd + n_ib],
        phi.immersed_upper,
    )
    assert torch.equal(packed[n_single + n_bnd + n_ib :], phi.immersed_lower)
