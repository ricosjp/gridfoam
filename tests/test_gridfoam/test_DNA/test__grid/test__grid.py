"""Tests for _grid/_grid module."""

from gridfoam.DNA._grid._grid import (
    NodeType,
    PyBBox,
    PyCubeCode,
    PyGrid,
    PyOctreeLevel,
    PyOctreeNode,
    RawIndexConversionMode,
    generate_grid_from_polydata,
)


def test_grid_module_imports():
    """Test that all grid module symbols can be imported."""
    assert NodeType is not None
    assert PyBBox is not None
    assert PyCubeCode is not None
    assert PyGrid is not None
    assert PyOctreeLevel is not None
    assert PyOctreeNode is not None
    assert RawIndexConversionMode is not None
    assert generate_grid_from_polydata is not None


def test_raw_index_conversion_mode():
    """Test RawIndexConversionMode enum values."""
    assert hasattr(RawIndexConversionMode, "WRAP")
    assert hasattr(RawIndexConversionMode, "CLAMP")
    assert hasattr(RawIndexConversionMode, "MIRROR")
    assert hasattr(RawIndexConversionMode, "BORDER")


def test_node_type():
    """Test NodeType enum values."""
    assert hasattr(NodeType, "LEAF")
    assert hasattr(NodeType, "GHOST_FROM_PARENT")
    assert hasattr(NodeType, "GHOST_FROM_CHILD")
