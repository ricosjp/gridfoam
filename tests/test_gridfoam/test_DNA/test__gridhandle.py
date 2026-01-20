"""Tests for _gridhandle module."""

import pytest

from gridfoam.DNA._gridhandle import IGridHandle


def test_igrid_handle_import():
    """Test that IGridHandle can be imported."""
    from gridfoam.DNA._gridhandle import IGridHandle

    assert IGridHandle is not None


def test_igrid_handle_is_abstract():
    """Test that IGridHandle is abstract."""
    with pytest.raises(TypeError):
        IGridHandle()


def test_igrid_handle_has_required_methods():
    """Test that IGridHandle has required abstract methods."""
    assert hasattr(IGridHandle, "iter_levels")
    assert hasattr(IGridHandle, "iter_leaf_on_level")
    assert hasattr(IGridHandle, "iter_gfp_on_level")
    assert hasattr(IGridHandle, "iter_gfc_on_level")
    assert hasattr(IGridHandle, "iter_all_leaves")
    assert hasattr(IGridHandle, "allocate_field")
    assert hasattr(IGridHandle, "allocate_equation")
    assert hasattr(IGridHandle, "update_fvmatrix")
    assert hasattr(IGridHandle, "sync_halo_at_depth")
    assert hasattr(IGridHandle, "sync_halo")
    assert hasattr(IGridHandle, "sync_gfp_at_depth")
    assert hasattr(IGridHandle, "sync_gfp")
    assert hasattr(IGridHandle, "sync_gfc_at_depth")
    assert hasattr(IGridHandle, "sync_gfc")
    assert hasattr(IGridHandle, "sync_all")
    assert hasattr(IGridHandle, "get_dx_at_depth")
    assert hasattr(IGridHandle, "grid")
    assert hasattr(IGridHandle, "mesh")
    assert hasattr(IGridHandle, "config")
