extern crate nalgebra as na;
use na::{matrix, SMatrix};

use crate::core::index::{CastRawIndex, RawIndexExt};
use crate::core::types::{IndexBounds, RawIndex, RawIndexConversionMode};

/// Precomputed direction vectors for all 27 neighbors in 3D space
///
/// This matrix contains all possible neighbor offsets in a 3x3x3 neighborhood.
/// Each column represents a direction vector (x, y, z) that can be added
/// to a grid position to get a neighbor position.
///
/// The directions are ordered in a specific pattern for efficient access
/// and consistent neighbor enumeration.
pub const DIRECTIONS: SMatrix<i64, 3, 27> = matrix![
    -1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1;
    -1,-1,-1, 0, 0, 0, 1, 1, 1,-1,-1,-1, 0, 0, 0, 1, 1, 1,-1,-1,-1, 0, 0, 0, 1, 1, 1;
    -1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1;
];

/// Named directions for common neighbor access
///
/// These enum values correspond to specific columns in the DIRECTIONS matrix,
/// providing named access to commonly used neighbor directions.
pub enum Direction {
    /// Negative Z direction (index 4 in DIRECTIONS matrix)
    ZMinus = 4,
    /// Negative Y direction (index 10 in DIRECTIONS matrix)
    YMinus = 10,
    /// Negative X direction (index 12 in DIRECTIONS matrix)
    XMinus = 12,
    /// Center position (index 13 in DIRECTIONS matrix)
    Center = 13,
    /// Positive X direction (index 14 in DIRECTIONS matrix)
    XPlus = 14,
    /// Positive Y direction (index 16 in DIRECTIONS matrix)
    YPlus = 16,
    /// Positive Z direction (index 22 in DIRECTIONS matrix)
    ZPlus = 22,
}

impl Direction {
    /// Get the offset vector for this direction
    ///
    /// Returns the 3D offset vector corresponding to this direction.
    /// The offset can be added to a grid position to get the neighbor position.
    ///
    /// # Returns
    ///
    /// A RawIndex containing the offset vector
    ///
    /// # Example
    ///
    /// ```
    /// let dir = Direction::XPlus;
    /// let offset = dir.offset();
    /// assert_eq!(offset, RawIndex::new(1, 0, 0));
    /// ```
    pub fn offset(self) -> RawIndex {
        DIRECTIONS.column(self as usize).into()
    }
}

/// Trait for finding neighbor indices in a 3D grid
///
/// This trait provides functionality to get all neighbors of a given grid position,
/// with configurable boundary handling and optional inclusion of the center position.
pub trait NeighborIndices {
    type Output;
    /// Get all neighbor indices for a given position
    ///
    /// Returns a vector of optional indices, where each element corresponds
    /// to a neighbor position. None values indicate out-of-bounds neighbors
    /// (depending on the boundary mode).
    ///
    /// # Arguments
    ///
    /// * `bounds` - The grid bounds defining valid ranges
    /// * `mode` - The boundary handling mode for out-of-bounds neighbors
    /// * `include_self` - Whether to include the center position (index 13)
    ///
    /// # Returns
    ///
    /// A vector of 27 optional indices, one for each possible neighbor
    ///
    /// # Example
    ///
    /// ```
    /// let index = Index::new(1, 1, 1);
    /// let bounds = Bounds::new(5, 5, 5);
    /// let neighbors = index.neighbor_indices(&bounds, IndexMode::Border, false);
    /// // Returns 27 optional indices, with None for out-of-bounds positions
    /// ```
    fn neighbor_indices(
        &self,
        bounds: &IndexBounds,
        mode: RawIndexConversionMode,
        include_self: bool,
    ) -> Vec<Option<Self::Output>>;
}

impl<T: CastRawIndex> NeighborIndices for T {
    type Output = T;
    fn neighbor_indices(
        &self,
        bounds: &IndexBounds,
        mode: RawIndexConversionMode,
        include_self: bool,
    ) -> Vec<Option<Self::Output>> {
        DIRECTIONS
            .column_iter()
            .enumerate()
            .map(|(i, dir)| {
                if !include_self && i == (Direction::Center as usize) {
                    None
                } else {
                    let coords: RawIndex = self.to_raw_index() + dir;
                    coords.convert_to::<T>(mode, bounds)
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{GlobalIndex, LocalIndex};
    use rstest::rstest;

    #[rstest]
    #[case::wrap(GlobalIndex::new(2, 2, 2), IndexBounds::new(3, 4, 5), RawIndexConversionMode::Wrap, true, vec![
        Some(GlobalIndex::new(1, 1, 1)), Some(GlobalIndex::new(2, 1, 1)),Some(GlobalIndex::new(0, 1, 1)),
        Some(GlobalIndex::new(1, 2, 1)), Some(GlobalIndex::new(2, 2, 1)),Some(GlobalIndex::new(0, 2, 1)),
        Some(GlobalIndex::new(1, 3, 1)), Some(GlobalIndex::new(2, 3, 1)),Some(GlobalIndex::new(0, 3, 1)),
        Some(GlobalIndex::new(1, 1, 2)), Some(GlobalIndex::new(2, 1, 2)),Some(GlobalIndex::new(0, 1, 2)),
        Some(GlobalIndex::new(1, 2, 2)), Some(GlobalIndex::new(2, 2, 2)),Some(GlobalIndex::new(0, 2, 2)),
        Some(GlobalIndex::new(1, 3, 2)), Some(GlobalIndex::new(2, 3, 2)),Some(GlobalIndex::new(0, 3, 2)),
        Some(GlobalIndex::new(1, 1, 3)), Some(GlobalIndex::new(2, 1, 3)),Some(GlobalIndex::new(0, 1, 3)),
        Some(GlobalIndex::new(1, 2, 3)), Some(GlobalIndex::new(2, 2, 3)),Some(GlobalIndex::new(0, 2, 3)),
        Some(GlobalIndex::new(1, 3, 3)), Some(GlobalIndex::new(2, 3, 3)),Some(GlobalIndex::new(0, 3, 3)),
        ])]
    #[case::clamp(GlobalIndex::new(2, 2, 2), IndexBounds::new(3, 4, 5), RawIndexConversionMode::Clamp, true, vec![
        Some(GlobalIndex::new(1, 1, 1)), Some(GlobalIndex::new(2, 1, 1)),Some(GlobalIndex::new(2, 1, 1)),
        Some(GlobalIndex::new(1, 2, 1)), Some(GlobalIndex::new(2, 2, 1)),Some(GlobalIndex::new(2, 2, 1)),
        Some(GlobalIndex::new(1, 3, 1)), Some(GlobalIndex::new(2, 3, 1)),Some(GlobalIndex::new(2, 3, 1)),
        Some(GlobalIndex::new(1, 1, 2)), Some(GlobalIndex::new(2, 1, 2)),Some(GlobalIndex::new(2, 1, 2)),
        Some(GlobalIndex::new(1, 2, 2)), Some(GlobalIndex::new(2, 2, 2)),Some(GlobalIndex::new(2, 2, 2)),
        Some(GlobalIndex::new(1, 3, 2)), Some(GlobalIndex::new(2, 3, 2)),Some(GlobalIndex::new(2, 3, 2)),
        Some(GlobalIndex::new(1, 1, 3)), Some(GlobalIndex::new(2, 1, 3)),Some(GlobalIndex::new(2, 1, 3)),
        Some(GlobalIndex::new(1, 2, 3)), Some(GlobalIndex::new(2, 2, 3)),Some(GlobalIndex::new(2, 2, 3)),
        Some(GlobalIndex::new(1, 3, 3)), Some(GlobalIndex::new(2, 3, 3)),Some(GlobalIndex::new(2, 3, 3)),
    ])]
    #[case::mirror(GlobalIndex::new(2, 2, 2), IndexBounds::new(3, 4, 5), RawIndexConversionMode::Mirror, true, vec![
        Some(GlobalIndex::new(1, 1, 1)), Some(GlobalIndex::new(2, 1, 1)),Some(GlobalIndex::new(1, 1, 1)),
        Some(GlobalIndex::new(1, 2, 1)), Some(GlobalIndex::new(2, 2, 1)),Some(GlobalIndex::new(1, 2, 1)),
        Some(GlobalIndex::new(1, 3, 1)), Some(GlobalIndex::new(2, 3, 1)),Some(GlobalIndex::new(1, 3, 1)),
        Some(GlobalIndex::new(1, 1, 2)), Some(GlobalIndex::new(2, 1, 2)),Some(GlobalIndex::new(1, 1, 2)),
        Some(GlobalIndex::new(1, 2, 2)), Some(GlobalIndex::new(2, 2, 2)),Some(GlobalIndex::new(1, 2, 2)),
        Some(GlobalIndex::new(1, 3, 2)), Some(GlobalIndex::new(2, 3, 2)),Some(GlobalIndex::new(1, 3, 2)),
        Some(GlobalIndex::new(1, 1, 3)), Some(GlobalIndex::new(2, 1, 3)),Some(GlobalIndex::new(1, 1, 3)),
        Some(GlobalIndex::new(1, 2, 3)), Some(GlobalIndex::new(2, 2, 3)),Some(GlobalIndex::new(1, 2, 3)),
        Some(GlobalIndex::new(1, 3, 3)), Some(GlobalIndex::new(2, 3, 3)),Some(GlobalIndex::new(1, 3, 3)),
        ])]
    #[case::border_include_self(GlobalIndex::new(2, 2, 2), IndexBounds::new(3, 4, 5), RawIndexConversionMode::Border, true, vec![
        Some(GlobalIndex::new(1, 1, 1)), Some(GlobalIndex::new(2, 1, 1)),None,
        Some(GlobalIndex::new(1, 2, 1)), Some(GlobalIndex::new(2, 2, 1)),None,
        Some(GlobalIndex::new(1, 3, 1)), Some(GlobalIndex::new(2, 3, 1)),None,
        Some(GlobalIndex::new(1, 1, 2)), Some(GlobalIndex::new(2, 1, 2)),None,
        Some(GlobalIndex::new(1, 2, 2)), Some(GlobalIndex::new(2, 2, 2)),None,
        Some(GlobalIndex::new(1, 3, 2)), Some(GlobalIndex::new(2, 3, 2)),None,
        Some(GlobalIndex::new(1, 1, 3)), Some(GlobalIndex::new(2, 1, 3)),None,
        Some(GlobalIndex::new(1, 2, 3)), Some(GlobalIndex::new(2, 2, 3)),None,
        Some(GlobalIndex::new(1, 3, 3)), Some(GlobalIndex::new(2, 3, 3)),None,
    ])]
    #[case::border_exclude_self(GlobalIndex::new(2, 2, 2), IndexBounds::new(3, 4, 5), RawIndexConversionMode::Border, false, vec![
        Some(GlobalIndex::new(1, 1, 1)), Some(GlobalIndex::new(2, 1, 1)),None,
        Some(GlobalIndex::new(1, 2, 1)), Some(GlobalIndex::new(2, 2, 1)),None,
        Some(GlobalIndex::new(1, 3, 1)), Some(GlobalIndex::new(2, 3, 1)),None,
        Some(GlobalIndex::new(1, 1, 2)), Some(GlobalIndex::new(2, 1, 2)),None,
        Some(GlobalIndex::new(1, 2, 2)), None, None,
        Some(GlobalIndex::new(1, 3, 2)), Some(GlobalIndex::new(2, 3, 2)),None,
        Some(GlobalIndex::new(1, 1, 3)), Some(GlobalIndex::new(2, 1, 3)),None,
        Some(GlobalIndex::new(1, 2, 3)), Some(GlobalIndex::new(2, 2, 3)),None,
        Some(GlobalIndex::new(1, 3, 3)), Some(GlobalIndex::new(2, 3, 3)),None,
    ])]
    fn test_neighbor_indices_global(
        #[case] input: GlobalIndex,
        #[case] bounds: IndexBounds,
        #[case] mode: RawIndexConversionMode,
        #[case] include_self: bool,
        #[case] expected: Vec<Option<GlobalIndex>>,
    ) {
        assert_eq!(
            input.neighbor_indices(&bounds, mode, include_self),
            expected
        );
    }

    #[rstest]
    #[case(LocalIndex::new(2, 2, 2), IndexBounds::new(3, 4, 5), RawIndexConversionMode::Border, false, vec![
        Some(LocalIndex::new(1, 1, 1)), Some(LocalIndex::new(2, 1, 1)),None,
        Some(LocalIndex::new(1, 2, 1)), Some(LocalIndex::new(2, 2, 1)),None,
        Some(LocalIndex::new(1, 3, 1)), Some(LocalIndex::new(2, 3, 1)),None,
        Some(LocalIndex::new(1, 1, 2)), Some(LocalIndex::new(2, 1, 2)),None,
        Some(LocalIndex::new(1, 2, 2)), None, None,
        Some(LocalIndex::new(1, 3, 2)), Some(LocalIndex::new(2, 3, 2)),None,
        Some(LocalIndex::new(1, 1, 3)), Some(LocalIndex::new(2, 1, 3)),None,
        Some(LocalIndex::new(1, 2, 3)), Some(LocalIndex::new(2, 2, 3)),None,
        Some(LocalIndex::new(1, 3, 3)), Some(LocalIndex::new(2, 3, 3)),None,
    ])]
    fn test_neighbor_indices_local(
        #[case] input: LocalIndex,
        #[case] bounds: IndexBounds,
        #[case] mode: RawIndexConversionMode,
        #[case] include_self: bool,
        #[case] expected: Vec<Option<LocalIndex>>,
    ) {
        assert_eq!(
            input.neighbor_indices(&bounds, mode, include_self),
            expected
        );
    }
}
