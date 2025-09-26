# cubion.pyi

import numpy as np
import pyvista as pv

from gridfoam._base._field import Field

DIRECTIONS: np.ndarray[tuple[int, int], np.dtype[np.int64]]
"""
Precomputed direction vectors for all 27 neighbors in 3D space.

This matrix contains all possible neighbor offsets in a 3x3x3 neighborhood.
Each row represents a direction vector (x, y, z) that can be added
to a grid position to get a neighbor position.

The directions are ordered in a specific pattern for efficient access
and consistent neighbor enumeration.

Shape: (27, 3)
Dtype: np.int64

Example:
    [-1, -1, -1]  # neighbor at (-1, -1, -1)
    [ 0, -1, -1]  # neighbor at ( 0, -1, -1)
    [ 1, -1, -1]  # neighbor at ( 1, -1, -1)
    [-1,  0, -1]  # neighbor at (-1,  0, -1)
    [ 0,  0, -1]  # neighbor at ( 0,  0, -1)
    ...
"""

def generate_grid_from_polydata(
    polydata: pv.PolyData, config_path: str
) -> PyGrid: ...
"""
Generate an octree grid from PyVista PolyData.

This function creates an octree grid structure from a PyVista PolyData object
using the specified configuration file. The function loads the mesh data,
applies the octree refinement strategy, and returns the resulting grid.

Parameters
----------
polydata : pv.PolyData
    PyVista PolyData object containing mesh geometry.
config_path : str
    Path to the YAML configuration file.

Returns
-------
PyGrid
    A PyGrid containing the complete octree structure.

Notes
-----
The configuration file path should be specified in the function call.
"""

class RawIndexConversionMode:
    """Enum for handling boundary conditions when converting indices."""

    WRAP: RawIndexConversionMode
    """Out-of-bounds coordinates are wrapped
    around the grid using modulo arithmetic."""

    CLAMP: RawIndexConversionMode
    """Out-of-bounds coordinates are clamped to the nearest boundary."""

    MIRROR: RawIndexConversionMode
    """Out-of-bounds coordinates are mirrored around the boundary."""

    BORDER: RawIndexConversionMode
    """Out-of-bounds coordinates return None."""

class NodeType:
    """Types of nodes in the octree structure."""

    LEAF: NodeType
    """Leaf node containing actual mesh intersection data."""

    GHOST_FROM_PARENT: NodeType
    """A ghost node retrieving data from the parent."""

    GHOST_FROM_CHILD: NodeType
    """A ghost node retrieving data from the child."""

class PyBBox:
    """Bounding box in 3D space."""

    lower: np.ndarray[np.float64]
    """Lower bounds of the bounding box [x_min, y_min, z_min]."""

    upper: np.ndarray[np.float64]
    """Upper bounds of the bounding box [x_max, y_max, z_max]."""

class PyCubeCode:
    """Cube code for identifying octree nodes in 3D space."""

    def to_global_index(self, depth: int) -> np.ndarray[np.uint64]: ...
    """
    Convert the cube code to a global index at the specified depth.

    Parameters
    ----------
    depth : int
        The depth level for the conversion.

    Returns
    -------
    np.ndarray[np.uint64]
        A NumPy array containing the global index coordinates [x, y, z].
    """

    def parent_and_offset_py(
        self, depth: int
    ) -> tuple[PyCubeCode, np.ndarray[np.uint32]]: ...
    """
    Get the parent cube code and local offset within the parent.

    Parameters
    ----------
    depth : int
        The current depth level.

    Returns
    -------
    tuple[PyCubeCode, np.ndarray[np.uint32]]
        A tuple containing the parent cube code and local offset as NumPy array.
    """

    def children(self, depth: int) -> list[PyCubeCode]: ...
    """
    Get the children cube codes at the next depth level.

    Parameters
    ----------
    depth : int
        The current depth level.

    Returns
    -------
    list[PyCubeCode]
        A list of 8 child cube codes.
    """

    def neighbor_codes(
        self,
        depth: int,
        bounds: np.ndarray[np.uint64],
        mode: RawIndexConversionMode,
        include_self: bool,
    ) -> list[PyCubeCode | None]: ...
    """
    Get the neighbor cube codes in a 3x3x3 neighborhood.

    This method returns all 27 neighbors
    (including the center cell if include_self=True)
    in a 3x3x3 neighborhood around the current cube code.

    Parameters
    ----------
    depth : int
        The current depth level (0 = root level).
    bounds : np.ndarray[np.uint64]
        Grid bounds as [max_x, max_y, max_z].
    mode : RawIndexConversionMode
        How to handle out-of-bounds neighbors.
    include_self : bool
        Whether to include the center cell itself.

    Returns
    -------
    list[PyCubeCode | None]
        List of 27 neighbor cube codes.
        None indicates out-of-bounds or invalid neighbors.
    """

    def value(self) -> int: ...
    """Get the raw cube code value as a 128-bit integer."""

class PyOctreeNode:
    """A single node in the octree structure."""

    cubecode: PyCubeCode
    """Unique identifier for this node in the octree hierarchy."""

    face_ids: np.ndarray[np.uint32]
    """Face IDs that intersect with this node's bounding box."""

    node_type: NodeType
    """Type of this node (leaf, ghost, etc.)."""

    cur: Field
    """Current field."""

    old: Field
    """Old field."""

class PyOctreeLevel:
    """A single level of the octree structure."""

    nodes: dict[PyCubeCode, PyOctreeNode]
    """Map of cube codes to nodes at this level."""

    bounds: np.ndarray[np.uint64]
    """Grid bounds for this level [max_x, max_y, max_z]."""

    depth: int
    """Depth level (0 = root)."""

    n_cells: int
    """Number of cells at this level."""

    n_cells_per_node: int
    """Number of cells per node at this level."""

class PyGrid:
    """A complete octree grid structure."""

    domain: PyBBox
    """Spatial domain of the octree."""

    blocksize: np.ndarray[np.uint64]
    """Block size at the root level [size_x, size_y, size_z]."""

    max_depth: int
    """Maximum depth reached during construction."""

    octree_levels: list[PyOctreeLevel]
    """All octree levels in the grid structure."""
