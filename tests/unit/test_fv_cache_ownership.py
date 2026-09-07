"""Tests for per-instance FV caches on grids and fields."""

from __future__ import annotations

import pathlib

from tests.helpers import channel_config, refined_grid

from gridfoam.core.field import CellField
from gridfoam.core.grid.factory import create_grid
from gridfoam.fv.boundary_ops import iter_boundary_batches
from gridfoam.fv.kernels.face_geometry import face_geometry
from gridfoam.meta.enums import FieldRole


def test_face_geometry_is_cached_on_grid_fv_cache():
    grid = refined_grid()
    geo1 = face_geometry(grid)
    assert grid.fv_cache.face_geometry is geo1
    assert face_geometry(grid) is geo1

    grid.invalidate_derived_caches()
    assert grid.fv_cache.face_geometry is None
    geo2 = face_geometry(grid)
    assert geo2 is not geo1
    assert grid.fv_cache.face_geometry is geo2


def test_boundary_batches_are_cached_on_field_fv_cache(
    tmp_path: pathlib.Path,
):
    grid = create_grid(channel_config(tmp_path))
    field = CellField(grid, "U", FieldRole.TRANSIENT, (3,))
    batches1 = tuple(iter_boundary_batches(field))
    assert len(batches1) > 0
    assert field.fv_cache.boundary_batches == batches1

    grid.invalidate_derived_caches()
    assert field.fv_cache.boundary_batches is None
    batches2 = tuple(iter_boundary_batches(field))
    assert field.fv_cache.boundary_batches == batches2
    assert batches2 is not batches1
