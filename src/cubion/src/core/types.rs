extern crate nalgebra as na;
use na::{SVector, Vector3};
use pyo3::pyclass;
use rstar::AABB;
use std::{collections::HashMap, hash::BuildHasherDefault};
use wyhash2::WyHash;

pub type WyHasher = BuildHasherDefault<WyHash>;

/// Types for the local index (32 bits at most for each axis)
pub type LocalIndexType = u32;
/// Types for the global index (42 bits at most for each axis)
pub type GlobalIndexType = u64;
/// Types for the raw index (42 bits + 1 bit for the sign at most for each axis)
pub type RawIndexType = i64;

/// Types for the cube code (128 bits)
pub type CubeCodeType = u128;
/// Types for the octree code (96 bits)
pub type OctreeCodeType = u128;
/// Types for the root code (30 bits)
pub type RootCodeType = u32;

/// Cube code type for identifying octree nodes
///
/// A 128-bit identifier that combines root code and octree code to uniquely
/// identify any node in the octree hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CubeCode(pub CubeCodeType);

/// Root code type for positioning at the root level of the octree
///
/// The available bit length for the root code is calculated as follows.
/// CUBE_CODE_BIT_LENGTH - 3 * MAX_OCTREE_DEPTH
/// 128 - 3 * 32 = 32
/// so we reserve 10 bits (1024) for each axis.
#[derive(Debug, Clone, PartialEq)]
pub struct RootCode(pub RootCodeType);

/// Octree code type for local octree positioning
///
/// A 128-bit code that represents the position within a specific octree,
/// using Morton encoding for efficient neighbor calculations.
#[derive(Debug, Clone, PartialEq)]
pub struct OctreeCode(pub OctreeCodeType);

/// Vector type for storing 8 child nodes in an octree
///
/// This type represents the 8 children of an octree node, arranged in a specific order
/// that corresponds to the octant positions in 3D space.
pub type ChildVector<T> = SVector<T, 8>;

/// Raw index type using signed integers for boundary calculations
///
/// This type allows negative values for neighbor calculations and boundary handling.
/// It's used internally for operations that may go outside the valid grid bounds.
pub type RawIndex = Vector3<RawIndexType>;

/// Local index type using unsigned integers for grid operations
///
/// This type represents indices relative to a specific cube's origin.
/// It's used for operations within a single cube's local coordinate system.
pub type LocalIndex = Vector3<LocalIndexType>;

/// Global index type using unsigned integers for global operations
///
/// This type represents indices relative to the entire grid's origin.
/// It's used for operations that require a full grid-wide reference.
pub type GlobalIndex = Vector3<GlobalIndexType>;

/// Bounds of the index space
///
/// This type represents the bounds of the index space.
/// It's used for operations that require a full grid-wide reference.
pub type IndexBounds = Vector3<GlobalIndexType>;

/// Enum for handling boundary conditions when converting from RawIndex to unsigned GlobalIndex or LocalIndex.
///
/// This enum defines different strategies for dealing with indices that fall outside the grid bounds
/// during conversion from a signed RawIndex to an unsigned index type (GlobalIndex or LocalIndex).
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RawIndexConversionMode {
    /// Out-of-bounds coordinates are wrapped around the grid using modulo arithmetic.
    ///
    /// This mode treats the grid as periodic, wrapping indices that exceed
    /// bounds to the opposite side of the grid.
    Wrap,

    /// Out-of-bounds coordinates are clamped to the nearest boundary.
    ///
    /// This mode clamps indices to the valid range [0, bounds-1].
    /// Useful for sampling operations where you want to stay within bounds.
    Clamp,

    /// Out-of-bounds coordinates are mirrored around the boundary.
    ///
    /// This mode treats the boundary as a mirror, reflecting indices that exceed bounds.
    /// Useful for operations where you want to maintain symmetry around the boundary.
    Mirror,

    /// Out-of-bounds coordinates return None.
    ///
    /// This mode returns None when an index is out of bounds, allowing
    /// the caller to handle the boundary condition explicitly.
    Border,
}

/// Point type representing 3D coordinates
pub type Point = [f64; 3];

/// Bounding box type
pub type BBox = AABB<Point>;

/// Types of nodes in the octree structure
///
/// Different node types serve different purposes in the octree hierarchy,
/// enabling efficient neighbor calculations and boundary handling.
#[pyclass(frozen)]
#[derive(Debug, Clone)]
pub enum NodeType {
    /// Leaf node containing actual mesh intersection data
    ///
    /// These nodes represent the finest level of the octree and contain
    /// the actual face IDs that intersect with the node's bounding box.
    Leaf,

    /// Ghost node created from child refinement
    ///
    /// These nodes are created when a neighbor is refined to a finer level.
    /// They provide boundary information for the refined neighbor.
    GhostFromChild,

    /// Ghost node created from parent coarsening
    ///
    /// These nodes are created when a neighbor remains at a coarser level.
    /// They provide boundary information for the coarser neighbor.
    GhostFromParent,
}

/// A single node in the octree structure
///
/// Represents a node in the octree hierarchy with its spatial identifier,
/// associated mesh faces, and node type classification.
pub struct OctreeNode {
    /// Unique identifier for this node in the octree hierarchy
    pub cubecode: CubeCode,
    /// Face IDs that intersect with this node's bounding box
    pub face_ids: Vec<usize>,
    /// Type of this node (leaf, ghost, etc.)
    pub node_type: NodeType,
}

/// A single level of the octree structure
///
/// Contains all nodes at a specific depth level and provides methods
/// for level-wise operations like splitting and ghost node generation.
pub struct OctreeLevel {
    /// Map of cube codes to nodes at this level
    pub nodes: HashMap<CubeCode, OctreeNode, WyHasher>,
    /// Grid bounds for this level
    pub bounds: IndexBounds,
    /// Depth level (0 = root)
    pub depth: usize,
}

// TODO: add more split strategies: curvature, etc.
/// Default split strategy for octree construction
///
/// This strategy splits nodes that are under the depth limit and have
/// intersecting mesh faces.
pub struct DefaultSplitStrategy {
    /// Maximum depth allowed for splitting
    pub depth_limit: usize,
    /// Refinement parameter for split criteria
    pub alpha: f64,
}
