use crate::core::constants::MAX_OCTREE_DEPTH;
use crate::core::types::{IndexBounds, OctreeCodeType, RawIndex};

#[derive(Debug, Clone, thiserror::Error, PartialEq)]
pub enum IndexError {
    #[error("Index overflow: {raw_index:?} exceeds maximum value for target type")]
    IndexOverflow { raw_index: RawIndex },

    #[error("Index underflow: {raw_index:?} is below minimum value for target type")]
    IndexUnderflow { raw_index: RawIndex },

    #[error("Raveled index {raveled_index} is out of bounds: {bounds:?}")]
    RaveledIndexOutOfBounds {
        raveled_index: OctreeCodeType,
        bounds: IndexBounds,
    },
}

#[derive(Debug, Clone, thiserror::Error, PartialEq)]
pub enum MortonError {
    #[error("Invalid depth: {depth} is over the maximum depth {MAX_OCTREE_DEPTH}")]
    InvalidDepth { depth: usize },
}
