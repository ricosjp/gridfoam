"""
Grid and field caches are owned by instances and refresh after geometry
invalidation.
"""

from __future__ import annotations

import pathlib

import torch
from tests.helpers import channel_config, refined_grid

from gridfoam.boundaries.utils import get_mask_and_size
from gridfoam.core.field import CellField
from gridfoam.core.grid.factory import create_grid
from gridfoam.fv.boundary_ops import iter_boundary_batches
from gridfoam.fv.kernels.face_geometry import face_geometry
from gridfoam.meta.enums import DomainBoundaryPatch, FieldRole


def test_face_geometry_is_cached_on_grid_fv_cache() -> None:
    """
    Face geometry is reused until invalidation, then rebuilt on the grid
    cache.
    """
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
) -> None:
    """
    Boundary batches live on the field cache and are repopulated after
    invalidation.
    """
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


def test_boundary_mask_size_and_area_refresh_after_geometry_invalidation(
    tmp_path: pathlib.Path,
) -> None:
    """
    Invalidation refreshes patch masks, face counts, area vectors, and
    magnitudes.
    """
    grid = create_grid(channel_config(tmp_path))
    field = CellField(grid, "U", FieldRole.LOCAL, (3,))
    patch = DomainBoundaryPatch.X_MINUS
    mask, count = get_mask_and_size(grid, patch)
    assert count == int(mask.sum())
    assert get_mask_and_size(grid, patch)[0] is mask
    batch = next(
        b for b in iter_boundary_batches(field) if b.patch_name == patch
    )
    torch.testing.assert_close(batch.Sf, grid.domain_bnd_Sf[mask])
    torch.testing.assert_close(
        batch.mag_Sf, torch.linalg.vector_norm(batch.Sf, dim=1)
    )

    grid.domain_bnd_Sf.mul_(2)
    grid.invalidate_derived_caches()
    new_mask, new_count = get_mask_and_size(grid, patch)
    assert new_mask is not mask
    assert new_count == count
    new_batch = next(
        b for b in iter_boundary_batches(field) if b.patch_name == patch
    )
    torch.testing.assert_close(new_batch.Sf, 2 * batch.Sf)
    torch.testing.assert_close(new_batch.mag_Sf, 2 * batch.mag_Sf)

    grid.domain_bnd_dir_id[mask] = (
        DomainBoundaryPatch.X_PLUS.to_direction().value
    )
    grid.invalidate_derived_caches()
    _, new_count = get_mask_and_size(grid, patch)
    assert new_count == 0
