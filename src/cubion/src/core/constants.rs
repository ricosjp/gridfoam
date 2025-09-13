/// Maximum depth allowed in the octree structure
///
/// This constant defines the maximum number of levels in the octree.
/// Each level doubles the resolution in each dimension.
pub const MAX_OCTREE_DEPTH: usize = 32;

/// Total bit length required for octree codes
///
/// Calculated as 3 bits per level (one for each dimension) times the maximum depth.
/// This determines how many bits are needed to encode the full octree position.
pub const OCTREE_CODE_BIT_LENGTH: usize = 3 * MAX_OCTREE_DEPTH;

/// Number of bits allocated per axis in the root code
///
/// Each axis gets 10 bits, allowing for 1024 different positions per axis
/// at the root level of the octree.
const ROOT_AXIS_BIT_LENGTH: usize = 10;

/// Total bit length for the root code
///
/// 3 axes × 10 bits per axis = 30 bits total for root positioning.
pub const ROOT_CODE_BIT_LENGTH: usize = 3 * ROOT_AXIS_BIT_LENGTH;

/// Maximum value for any axis at the root level
///
/// This is 2^10 = 1024, representing the maximum coordinate
/// in any dimension at the root level.
pub const MAX_BLOCK_AXIS: u64 = 1 << ROOT_AXIS_BIT_LENGTH as u64;
