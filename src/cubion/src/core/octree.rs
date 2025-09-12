use crate::core::{index::*, mesh::*, morton::*};
use crate::io::config::{Config, SplitStrategyType};
use pyo3::pyclass;
use std::{
    collections::{HashMap, HashSet},
    hash::BuildHasherDefault,
};

use wyhash2::WyHash;
type WyHasher = BuildHasherDefault<WyHash>;

#[derive(Debug, Clone, thiserror::Error, PartialEq)]
pub enum OctreeError {
    #[error("Invalid depth: {depth} is over the maximum depth {MAX_OCTREE_DEPTH}")]
    InvalidDepth { depth: usize },

    #[error("Raveled index {raveled_index} is out of bounds: {bounds:?}")]
    RaveledIndexOutOfBounds {
        raveled_index: u64,
        bounds: IndexBounds,
    },
}

/// Types of nodes in the octree structure
///
/// Different node types serve different purposes in the octree hierarchy,
/// enabling efficient neighbor calculations and boundary handling.
#[pyclass]
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

impl OctreeLevel {
    /// Create a new octree level
    ///
    /// # Arguments
    ///
    /// * `nodes` - Map of nodes at this level
    /// * `bounds` - Grid bounds for this level
    /// * `depth` - Depth level
    ///
    /// # Returns
    ///
    /// A new octree level with the specified properties
    fn new(
        nodes: HashMap<CubeCode, OctreeNode, WyHasher>,
        bounds: IndexBounds,
        depth: usize,
    ) -> Self {
        Self {
            nodes,
            bounds,
            depth,
        }
    }

    /// Collect cube codes that should be split based on the split strategy
    ///
    /// This method identifies nodes that need to be refined and collects
    /// their cube codes along with their neighbors for proper boundary handling.
    ///
    /// # Arguments
    ///
    /// * `split_strategy` - Strategy determining which nodes to split
    ///
    /// # Returns
    ///
    /// A set of cube codes that should be split
    fn collect_split_cubecodes(
        &self,
        split_strategy: &dyn SplitStrategy,
    ) -> HashSet<CubeCode, WyHasher> {
        let mut split_cubecode_set: HashSet<CubeCode, WyHasher> = HashSet::default();
        self.nodes
            .iter()
            .filter(|(_, node)| split_strategy.should_split(self.depth, node))
            .for_each(|(cubecode, _)| {
                split_cubecode_set.extend(
                    cubecode
                        .to_global_index(self.depth)
                        .neighbor_indices(&self.bounds, RawIndexConversionMode::Border, true)
                        .into_iter()
                        .flatten()
                        .map(|index| index.to_cubecode(self.depth)),
                );
            });
        split_cubecode_set
    }

    /// Collect ghost nodes for proper boundary handling
    ///
    /// Ghost nodes are created to maintain proper boundary conditions
    /// between nodes at different refinement levels.
    ///
    /// # Arguments
    ///
    /// * `split_cubecode_set` - Set of cube codes that are being split
    ///
    /// # Returns
    ///
    /// A map of ghost nodes with their cube codes
    fn collect_ghost_nodes(
        &self,
        split_cubecode_set: &HashSet<CubeCode, WyHasher>,
    ) -> HashMap<CubeCode, OctreeNode, WyHasher> {
        let mut neighbor_cubecode_set: HashSet<CubeCode, WyHasher> = HashSet::default();
        self.nodes.iter().for_each(|(cubecode, _)| {
            cubecode
                .to_global_index(self.depth)
                .neighbor_indices(&self.bounds, RawIndexConversionMode::Border, false)
                .into_iter()
                .flatten()
                .for_each(|index| {
                    let neighbor_cubecode = index.to_cubecode(self.depth);
                    neighbor_cubecode_set.insert(neighbor_cubecode);
                });
        });

        let mut ghost_nodes: HashMap<CubeCode, OctreeNode, WyHasher> = HashMap::default();
        neighbor_cubecode_set
            .into_iter()
            .filter(|cubecode| !self.nodes.contains_key(cubecode))
            .for_each(|cubecode| {
                if split_cubecode_set.contains(&cubecode) {
                    ghost_nodes.insert(
                        cubecode,
                        OctreeNode {
                            cubecode,
                            face_ids: Vec::new(),
                            node_type: NodeType::GhostFromChild,
                        },
                    );
                } else {
                    ghost_nodes.insert(
                        cubecode,
                        OctreeNode {
                            cubecode,
                            face_ids: Vec::new(),
                            node_type: NodeType::GhostFromParent,
                        },
                    );
                }
            });
        ghost_nodes
    }

    /// Collect child nodes created by splitting
    ///
    /// This method creates the child nodes that result from splitting
    /// parent nodes, including mesh intersection calculations.
    ///
    /// # Arguments
    ///
    /// * `split_cubecode_set` - Set of cube codes being split
    /// * `domain` - Spatial domain for AABB calculations
    /// * `mesh` - Triangle mesh for intersection calculations
    ///
    /// # Returns
    ///
    /// A map of child nodes with their cube codes
    fn collect_splitted_nodes(
        &self,
        split_cubecode_set: &HashSet<CubeCode, WyHasher>,
        domain: &BBox,
        mesh: &TriangleMesh,
    ) -> HashMap<CubeCode, OctreeNode, WyHasher> {
        let child_bounds = self.bounds * 2;
        let mut splitted: HashMap<CubeCode, OctreeNode, WyHasher> = HashMap::default();
        split_cubecode_set.into_iter().for_each(|cubecode| {
            cubecode
                .children(self.depth)
                .unwrap()
                .into_iter()
                .for_each(|&child_cubecode| {
                    let child_cubecode = CubeCode(child_cubecode);
                    let child_depth = self.depth + 1;
                    let bbox =
                        domain.calculate_bbox_from(&child_cubecode, &child_bounds, child_depth);
                    let child_face_ids = mesh.find_intersecting_face_ids(bbox);
                    splitted.insert(
                        child_cubecode,
                        OctreeNode {
                            cubecode: child_cubecode,
                            face_ids: child_face_ids,
                            node_type: NodeType::Leaf,
                        },
                    );
                });
        });
        splitted
    }

    /// Split this level to create the next level of refinement
    ///
    /// This method performs the complete splitting process for this level,
    /// including collecting split codes, creating ghost nodes, and generating
    /// child nodes.
    ///
    /// # Arguments
    ///
    /// * `domain` - Spatial domain for calculations
    /// * `mesh` - Triangle mesh for intersection calculations
    /// * `split_strategy` - Strategy for determining which nodes to split
    /// * `split_context` - Context for split decisions
    ///
    /// # Returns
    ///
    /// A new octree level containing the child nodes
    fn split_this_level(
        &mut self,
        domain: &BBox,
        mesh: &TriangleMesh,
        split_strategy: &dyn SplitStrategy,
    ) -> OctreeLevel {
        // collect split cubecodes
        let split_cubecode_set: HashSet<CubeCode, WyHasher> =
            self.collect_split_cubecodes(split_strategy);

        // remove nodes that are split
        self.nodes.retain(|k, _| !split_cubecode_set.contains(k));

        // collect ghost nodes
        let ghost_nodes: HashMap<CubeCode, OctreeNode, WyHasher> =
            self.collect_ghost_nodes(&split_cubecode_set);

        // add ghost nodes
        self.nodes.extend(ghost_nodes);

        // generate split nodes
        let splitted: HashMap<CubeCode, OctreeNode, WyHasher> =
            self.collect_splitted_nodes(&split_cubecode_set, domain, mesh);

        OctreeLevel::new(splitted, self.bounds * 2, self.depth + 1)
    }
}

// TODO: add more split strategies: curvature, etc.
/// Default split strategy for octree construction
///
/// This strategy splits nodes that are under the depth limit and have
/// intersecting mesh faces.
pub struct DefaultSplitStrategy {
    /// Maximum depth allowed for splitting
    depth_limit: usize,
    /// Refinement parameter for split criteria
    alpha: f64,
}

/// Trait for determining which nodes should be split
///
/// Different split strategies can be implemented to control
/// how the octree is refined based on various criteria.
pub trait SplitStrategy {
    /// Determine if a node should be split
    ///
    /// # Arguments
    ///
    /// * `node` - The node to evaluate
    /// * `context` - Context information for the decision
    ///
    /// # Returns
    ///
    /// `true` if the node should be split, `false` otherwise
    fn should_split(&self, depth: usize, node: &OctreeNode) -> bool;
}

impl SplitStrategy for DefaultSplitStrategy {
    fn should_split(&self, depth: usize, node: &OctreeNode) -> bool {
        let is_under_depth_limit = depth < self.depth_limit;
        let has_face_ids = !node.face_ids.is_empty();
        is_under_depth_limit && has_face_ids
    }
}

/// Builder for constructing octrees with configurable split strategies
///
/// This builder provides a flexible way to construct octrees using
/// different split strategies and parameters.
pub struct OctreeBuilder {
    /// Spatial domain for the octree
    domain: BBox,
    /// Block size at the root level
    blocksize: IndexBounds,
    /// Strategy for determining which nodes to split
    split_strategy: Box<dyn SplitStrategy>,
    /// Maximum depth allowed in the octree
    depth_limit: usize,
}

impl OctreeBuilder {
    /// Create a new octree builder
    ///
    /// # Arguments
    ///
    /// * `setting` - Grid settings containing domain and parameters
    /// * `split_strategy` - Strategy for determining which nodes to split
    ///
    /// # Returns
    ///
    /// A new octree builder with the specified configuration
    pub fn new(config: Config) -> Self {
        let refinement_config = config.octree.refinement;
        let split_strategy: Box<dyn SplitStrategy> = match refinement_config.strategy {
            SplitStrategyType::Default => Box::new(DefaultSplitStrategy {
                depth_limit: refinement_config.depth_limit,
                alpha: refinement_config.alpha,
            }),
        };
        Self {
            domain: config.grid.domain,
            blocksize: config.grid.blocksize,
            split_strategy,
            depth_limit: refinement_config.depth_limit,
        }
    }

    /// Build the octree from a triangle mesh
    ///
    /// This method performs the complete octree construction process,
    /// starting from the root level and refining based on the split strategy.
    ///
    /// # Arguments
    ///
    /// * `mesh` - The triangle mesh to build the octree from
    ///
    /// # Returns
    ///
    /// A complete octree grid structure
    pub fn build_with_mesh(&self, mesh: TriangleMesh) -> Grid {
        let mut octree_levels = Vec::<OctreeLevel>::new();
        octree_levels.push(self.generate_roots(&mesh));

        for depth in 0..self.depth_limit {
            let octree_level = octree_levels.get_mut(depth).unwrap();
            let new_octree_level =
                octree_level.split_this_level(&self.domain, &mesh, self.split_strategy.as_ref());
            if new_octree_level.nodes.is_empty() {
                break;
            }
            octree_levels.push(new_octree_level);
        }
        let max_depth: usize = octree_levels.len();
        Grid::new(self.domain, self.blocksize, max_depth, octree_levels, mesh)
    }

    /// Generate the root level of the octree
    ///
    /// Creates the initial level of the octree by dividing the domain
    /// into blocks and calculating mesh intersections.
    ///
    /// # Arguments
    ///
    /// * `mesh` - The triangle mesh for intersection calculations
    ///
    /// # Returns
    ///
    /// The root level of the octree
    fn generate_roots(&self, mesh: &TriangleMesh) -> OctreeLevel {
        let root_indices: Vec<LocalIndex> = generate_grid_indices(&self.blocksize);
        let mut nodes: HashMap<CubeCode, OctreeNode, WyHasher> = HashMap::default();

        root_indices.iter().for_each(|index| {
            let cubecode = CubeCode::new(index.to_root_code(), OctreeCode(0));
            let bbox = self
                .domain
                .calculate_bbox_from(&cubecode, &self.blocksize, 0);
            let face_ids = mesh.find_intersecting_face_ids(bbox);
            nodes.insert(
                cubecode,
                OctreeNode {
                    cubecode,
                    face_ids,
                    node_type: NodeType::Leaf,
                },
            );
        });
        OctreeLevel::new(nodes, self.blocksize, 0)
    }
}

/// Complete octree grid structure
///
/// Contains all levels of the octree in a compact, efficient representation
/// suitable for querying and traversal.
pub struct Grid {
    /// Spatial domain of the octree
    pub domain: BBox,
    /// Block size at the root level
    pub blocksize: IndexBounds,
    /// Maximum depth reached during construction
    pub max_depth: usize,
    /// All octree levels in compact representation
    pub octree_levels: Vec<OctreeLevel>,
    /// Triangle mesh used for intersection calculations
    pub mesh: TriangleMesh,
}

impl Grid {
    /// Create a new grid structure
    ///
    /// # Arguments
    ///
    /// * `domain` - Spatial domain of the octree
    /// * `blocksize` - Block size at the root level
    /// * `max_depth` - Maximum depth reached
    /// * `octree_levels` - All levels in octree representation
    /// * `mesh` - Mesh
    ///
    /// # Returns
    ///
    /// A new grid structure
    fn new(
        domain: BBox,
        blocksize: IndexBounds,
        max_depth: usize,
        octree_levels: Vec<OctreeLevel>,
        mesh: TriangleMesh,
    ) -> Self {
        Self {
            domain,
            blocksize,
            max_depth,
            octree_levels,
            mesh,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::{fixture, rstest};
    use std::path::Path;

    #[fixture]
    fn simple_mesh() -> TriangleMesh {
        TriangleMesh::from_stl("../../tests/data/stl/bunny.stl")
    }

    #[fixture]
    fn config() -> Config {
        let filepath = Path::new("../../tests/data/yaml/bunny.yaml");
        Config::from_file(filepath).unwrap()
    }

    #[rstest]

    fn test_build_grid(simple_mesh: TriangleMesh, config: Config) {
        let builder = OctreeBuilder::new(config);
        let grid = builder.build_with_mesh(simple_mesh);
        assert_eq!(grid.domain.lower(), [-4.0, -2.0, -2.0]);
        assert_eq!(grid.domain.upper(), [4.0, 2.0, 2.0]);
        assert_eq!(grid.blocksize, Vector3::<u64>::new(8, 4, 4));
        assert_eq!(grid.max_depth, 7);
    }
}
