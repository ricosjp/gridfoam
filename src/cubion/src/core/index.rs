extern crate nalgebra as na;
use na::{matrix, SMatrix, Vector3};

/// Raw index type using signed integers for boundary calculations
///
/// This type allows negative values for neighbor calculations and boundary handling.
/// It's used internally for operations that may go outside the valid grid bounds.
pub type RawIndex = Vector3<i64>;

/// Local index type using unsigned integers for grid operations
///
/// This type represents indices relative to a specific cube's origin.
/// It's used for operations within a single cube's local coordinate system.
pub type LocalIndex = Vector3<u32>;

/// Global index type using unsigned integers for global operations
///
/// This type represents indices relative to the entire grid's origin.
/// It's used for operations that require a full grid-wide reference.
pub type GlobalIndex = Vector3<u64>;

/// Bounds of the index space
///
/// This type represents the bounds of the index space.
/// It's used for operations that require a full grid-wide reference.
pub type IndexBounds = Vector3<u64>;

#[derive(Debug, Clone, thiserror::Error, PartialEq)]
pub enum IndexError {
    #[error("Invalid index: {raw_index:?} is out of bounds")]
    InvalidIndex { raw_index: RawIndex },

    #[error("Index overflow: {raw_index:?} exceeds maximum value for target type")]
    IndexOverflow { raw_index: RawIndex },

    #[error("Index underflow: {raw_index:?} is below minimum value for target type")]
    IndexUnderflow { raw_index: RawIndex },

    #[error("Raveled index {raveled_index} is out of bounds: {bounds:?}")]
    RaveledIndexOutOfBounds {
        raveled_index: u64,
        bounds: IndexBounds,
    },
}

/// Enum for handling boundary conditions when converting from RawIndex to unsigned GlobalIndex or LocalIndex.
///
/// This enum defines different strategies for dealing with indices that fall outside the grid bounds
/// during conversion from a signed RawIndex to an unsigned index type (GlobalIndex or LocalIndex).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RawIndexConversionMode {
    /// Raise an error when index is out of bounds
    ///
    /// This mode will panic if an index falls outside the valid grid bounds.
    /// Use this when you want to catch boundary violations immediately.
    Raise,
    /// Wrap the index to the other side of the grid
    ///
    /// This mode treats the grid as periodic, wrapping indices that exceed
    /// bounds to the opposite side of the grid.
    Wrap,
    /// Clip the index to the grid bounds
    ///
    /// This mode clamps indices to the valid range [0, bounds-1].
    /// Useful for sampling operations where you want to stay within bounds.
    Clip,
    /// Return None when index is out of bounds
    ///
    /// This mode returns None when an index is out of bounds, allowing
    /// the caller to handle the boundary condition explicitly.
    Border,
}

pub trait RawIndexConversion {
    fn convert<T>(self, mode: RawIndexConversionMode, bounds: &IndexBounds) -> Option<T>
    where
        T: CastRawIndex;
}

impl RawIndexConversion for RawIndex {
    fn convert<T>(self, mode: RawIndexConversionMode, bounds: &IndexBounds) -> Option<T>
    where
        T: CastRawIndex,
    {
        match mode {
            RawIndexConversionMode::Raise => {
                let valid = self.is_in_bounds(bounds);
                match valid {
                    true => Some(T::try_from_raw_index(&self).unwrap()),
                    false => panic!("Error: {}", IndexError::InvalidIndex { raw_index: self }),
                }
            }
            RawIndexConversionMode::Wrap => {
                let bounds = bounds.cast::<i64>();
                let wrapped = self.zip_map(&bounds, |i, b| (i % b + b) % b);
                Some(T::try_from_raw_index(&wrapped).unwrap())
            }
            RawIndexConversionMode::Clip => {
                let bounds = bounds.cast::<i64>();
                let clipped = self.zip_map(&bounds, |i, b| i.clamp(0, b - 1));
                Some(T::try_from_raw_index(&clipped).unwrap())
            }
            RawIndexConversionMode::Border => {
                let valid = self.is_in_bounds(bounds);
                valid.then(|| T::try_from_raw_index(&self).unwrap())
            }
        }
    }
}

pub trait CastRawIndex: Sized {
    fn try_from_raw_index(raw_index: &RawIndex) -> Result<Self, IndexError>;
    fn to_raw_index(&self) -> RawIndex;
}

impl CastRawIndex for GlobalIndex {
    fn try_from_raw_index(raw_index: &RawIndex) -> Result<Self, IndexError> {
        // Check for negative values before conversion
        if raw_index.x < 0 || raw_index.y < 0 || raw_index.z < 0 {
            return Err(IndexError::IndexUnderflow {
                raw_index: *raw_index,
            });
        }

        // Use try_cast to convert to u64
        raw_index
            .try_cast::<u64>()
            .ok_or(IndexError::IndexOverflow {
                raw_index: *raw_index,
            })
    }

    fn to_raw_index(&self) -> RawIndex {
        self.cast::<i64>()
    }
}

impl CastRawIndex for LocalIndex {
    fn try_from_raw_index(raw_index: &RawIndex) -> Result<Self, IndexError> {
        // Check for negative values before conversion
        if raw_index.x < 0 || raw_index.y < 0 || raw_index.z < 0 {
            return Err(IndexError::IndexUnderflow {
                raw_index: *raw_index,
            });
        }

        // Use try_cast to convert to u32
        raw_index
            .try_cast::<u32>()
            .ok_or(IndexError::IndexOverflow {
                raw_index: *raw_index,
            })
    }

    fn to_raw_index(&self) -> RawIndex {
        self.cast::<i64>()
    }
}

pub trait IsInBounds {
    fn is_in_bounds(&self, bounds: &IndexBounds) -> bool;
}

impl IsInBounds for RawIndex {
    fn is_in_bounds(&self, bounds: &IndexBounds) -> bool {
        self.x >= 0
            && self.x < bounds.x as i64
            && self.y >= 0
            && self.y < bounds.y as i64
            && self.z >= 0
            && self.z < bounds.z as i64
    }
}

impl IsInBounds for GlobalIndex {
    fn is_in_bounds(&self, bounds: &IndexBounds) -> bool {
        self.x < bounds.x && self.y < bounds.y && self.z < bounds.z
    }
}

impl IsInBounds for LocalIndex {
    fn is_in_bounds(&self, bounds: &IndexBounds) -> bool {
        self.x < bounds.x as u32 && self.y < bounds.y as u32 && self.z < bounds.z as u32
    }
}

/// Trait for converting multi-dimensional coordinates to flat indices
///
/// This trait provides methods to convert 3D grid coordinates to 1D array indices,
/// which is useful for storing grid data in flat arrays.
pub trait RavelMultiIndex {
    /// Convert multi-dimensional coordinates to a flat index
    ///
    /// Uses row-major ordering: index = x + y * x_stride + z * xy_stride
    ///
    /// # Arguments
    ///
    /// * `bounds` - The grid bounds defining the stride calculations
    ///
    /// # Returns
    ///
    /// The flat index corresponding to the multi-dimensional coordinates
    ///
    /// # Example
    ///
    /// ```
    /// let index = LocalIndex::new(1, 2, 0);
    /// let bounds = IndexBounds::new(3, 4, 5);
    /// let flat = index.ravel_multi_index(&bounds);
    /// // flat = 1 + 2*3 + 0*3*4 = 7
    /// ```
    fn ravel_multi_index(&self, bounds: &IndexBounds) -> u64;
}

impl RavelMultiIndex for LocalIndex {
    fn ravel_multi_index(&self, bounds: &IndexBounds) -> u64 {
        self.x as u64 + self.y as u64 * bounds.x + self.z as u64 * bounds.x * bounds.y
    }
}

impl RavelMultiIndex for GlobalIndex {
    fn ravel_multi_index(&self, bounds: &IndexBounds) -> u64 {
        self.x + self.y * bounds.x + self.z * bounds.x * bounds.y
    }
}

/// Trait for converting flat indices back to multi-dimensional coordinates
///
/// This trait provides the inverse operation of raveling, converting
/// 1D array indices back to 3D grid coordinates.
pub trait UnravelIndex {
    type Output;
    /// Convert a flat index to multi-dimensional coordinates
    ///
    /// Uses the inverse of row-major ordering to reconstruct 3D coordinates
    /// from a flat index.
    ///
    /// # Arguments
    ///
    /// * `bounds` - The grid bounds used for the original raveling
    ///
    /// # Returns
    ///
    /// The multi-dimensional coordinates corresponding to the flat index
    ///
    /// # Panics
    ///
    /// Panics if the flat index is out of bounds for the given grid size
    ///
    /// # Example
    ///
    /// ```
    /// let flat_index = 7u64;
    /// let bounds = Bounds::new(3, 4, 5);
    /// let coords = flat_index.unravel_index(&bounds);
    /// // coords = Index::new(1, 2, 0)
    /// ```
    fn unravel_index(self, bounds: &IndexBounds) -> Self::Output;
}

impl UnravelIndex for u64 {
    type Output = Result<LocalIndex, IndexError>;
    fn unravel_index(self, bounds: &IndexBounds) -> Self::Output {
        let total_size = bounds.size();
        if self >= total_size {
            return Err(IndexError::RaveledIndexOutOfBounds {
                raveled_index: self,
                bounds: *bounds,
            });
        }

        let x = self % bounds.x;
        let y = (self / bounds.x) % bounds.y;
        let z = self / (bounds.x * bounds.y);
        Ok(LocalIndex::new(x as u32, y as u32, z as u32))
    }
}

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
                    coords.convert::<T>(mode, bounds)
                }
            })
            .collect()
    }
}

/// Generate all valid grid indices for a given grid size
///
/// This function creates a vector containing all valid grid coordinates
/// for a grid of the specified size, ordered in row-major order.
///
/// # Arguments
///
/// * `bounds` - The grid bounds defining the grid size
///
/// # Returns
///
/// A vector containing all valid grid indices
///
/// # Example
///
/// ```
/// let bounds = IndexBounds::new(2, 2, 2);
/// let indices = generate_grid_indices(&bounds);
/// // Returns: [(0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1), (1,0,1), (0,1,1), (1,1,1)]
/// ```
pub fn generate_grid_indices(bounds: &IndexBounds) -> Vec<LocalIndex> {
    let [nx, ny, nz] = [bounds.x, bounds.y, bounds.z];
    let mut result = Vec::with_capacity((nx * ny * nz) as usize);

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                result.push(LocalIndex::new(x as u32, y as u32, z as u32));
            }
        }
    }
    result
}

trait Size {
    fn size(&self) -> u64;
}

impl Size for IndexBounds {
    fn size(&self) -> u64 {
        self.x * self.y * self.z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[should_panic]
    #[case::panic(
        RawIndex::new(3, 1, 4),
        RawIndexConversionMode::Raise,
        IndexBounds::new(6, 7, 4),
        Some(LocalIndex::new(3, 1, 4))
    )]
    #[case::no_panic(
        RawIndex::new(3, 1, 4),
        RawIndexConversionMode::Wrap,
        IndexBounds::new(6, 7, 4),
        Some(LocalIndex::new(3, 1, 0))
    )]
    #[case::no_panic(
        RawIndex::new(3, 1, 4),
        RawIndexConversionMode::Clip,
        IndexBounds::new(6, 7, 4),
        Some(LocalIndex::new(3, 1, 3))
    )]
    #[case::no_panic(
        RawIndex::new(3, 1, 4),
        RawIndexConversionMode::Border,
        IndexBounds::new(6, 7, 4),
        None
    )]
    fn test_index_conversion_local(
        #[case] input: RawIndex,
        #[case] mode: RawIndexConversionMode,
        #[case] bounds: IndexBounds,
        #[case] expected: Option<LocalIndex>,
    ) {
        assert_eq!(input.convert::<LocalIndex>(mode, &bounds), expected);
    }

    #[rstest]
    #[should_panic]
    #[case::panic(
        RawIndex::new(3, 1, 4),
        RawIndexConversionMode::Raise,
        IndexBounds::new(6, 7, 4),
        Some(GlobalIndex::new(3, 1, 4))
    )]
    #[case::no_panic(
        RawIndex::new(3, 1, 4),
        RawIndexConversionMode::Wrap,
        IndexBounds::new(6, 7, 4),
        Some(GlobalIndex::new(3, 1, 0))
    )]
    #[case::no_panic(
        RawIndex::new(3, 1, 4),
        RawIndexConversionMode::Clip,
        IndexBounds::new(6, 7, 4),
        Some(GlobalIndex::new(3, 1, 3))
    )]
    #[case::no_panic(
        RawIndex::new(3, 1, 4),
        RawIndexConversionMode::Border,
        IndexBounds::new(6, 7, 4),
        None
    )]
    fn test_index_conversion_global(
        #[case] input: RawIndex,
        #[case] mode: RawIndexConversionMode,
        #[case] bounds: IndexBounds,
        #[case] expected: Option<GlobalIndex>,
    ) {
        assert_eq!(input.convert::<GlobalIndex>(mode, &bounds), expected);
    }

    #[rstest]
    #[case(LocalIndex::new(0, 1, 2), IndexBounds::new(1, 2, 3), 0+1*1+2*1*2)]
    #[case(LocalIndex::new(3, 1, 2), IndexBounds::new(6, 7, 4), 3+1*6+2*6*7)]
    fn test_ravel_multi_index_local(
        #[case] input: LocalIndex,
        #[case] bounds: IndexBounds,
        #[case] expected: u64,
    ) {
        assert_eq!(input.ravel_multi_index(&bounds), expected);
    }

    #[rstest]
    #[case(GlobalIndex::new(0, 1, 2), IndexBounds::new(1, 2, 3), 0+1*1+2*1*2)]
    #[case(GlobalIndex::new(3, 1, 2), IndexBounds::new(6, 7, 4), 3+1*6+2*6*7)]
    fn test_ravel_multi_index_global(
        #[case] input: GlobalIndex,
        #[case] bounds: IndexBounds,
        #[case] expected: u64,
    ) {
        assert_eq!(input.ravel_multi_index(&bounds), expected);
    }

    #[rstest]
    #[case::no_panic(12, IndexBounds::new(3, 4, 5), Ok(LocalIndex::new(0, 0, 1)))]
    #[case::no_panic(27, IndexBounds::new(3, 4, 5), Ok(LocalIndex::new(0, 1, 2)))]
    #[case::no_panic(43, IndexBounds::new(3, 4, 5), Ok(LocalIndex::new(1, 2, 3)))]
    #[case::no_panic(75, IndexBounds::new(3, 4, 5), Err(IndexError::RaveledIndexOutOfBounds { raveled_index: 75, bounds: IndexBounds::new(3, 4, 5) }))]
    fn test_unravel_index(
        #[case] input: u64,
        #[case] bounds: IndexBounds,
        #[case] expected: Result<LocalIndex, IndexError>,
    ) {
        assert_eq!(input.unravel_index(&bounds), expected);
    }

    #[rstest]
    #[should_panic]
    #[case::panic(GlobalIndex::new(2, 2, 2), IndexBounds::new(3, 4, 5), RawIndexConversionMode::Raise, true, vec![
        Some(GlobalIndex::new(1, 1, 1)), Some(GlobalIndex::new(2, 1, 1)),Some(GlobalIndex::new(3, 1, 1)),
        Some(GlobalIndex::new(1, 2, 1)), Some(GlobalIndex::new(2, 2, 1)),Some(GlobalIndex::new(3, 2, 1)),
        Some(GlobalIndex::new(1, 3, 1)), Some(GlobalIndex::new(2, 3, 1)),Some(GlobalIndex::new(3, 3, 1)),
        Some(GlobalIndex::new(1, 1, 2)), Some(GlobalIndex::new(2, 1, 2)),Some(GlobalIndex::new(3, 1, 2)),
        Some(GlobalIndex::new(1, 2, 2)), Some(GlobalIndex::new(2, 2, 2)),Some(GlobalIndex::new(3, 2, 2)),
        Some(GlobalIndex::new(1, 3, 2)), Some(GlobalIndex::new(2, 3, 2)),Some(GlobalIndex::new(3, 3, 2)),
        Some(GlobalIndex::new(1, 1, 3)), Some(GlobalIndex::new(2, 1, 3)),Some(GlobalIndex::new(3, 1, 3)),
        Some(GlobalIndex::new(1, 2, 3)), Some(GlobalIndex::new(2, 2, 3)),Some(GlobalIndex::new(3, 2, 3)),
        Some(GlobalIndex::new(1, 3, 3)), Some(GlobalIndex::new(2, 3, 3)),Some(GlobalIndex::new(3, 3, 3)),
    ])]
    #[case::no_panic(GlobalIndex::new(2, 2, 2), IndexBounds::new(3, 4, 5), RawIndexConversionMode::Wrap, true, vec![
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
    #[case::no_panic(GlobalIndex::new(2, 2, 2), IndexBounds::new(3, 4, 5), RawIndexConversionMode::Clip, true, vec![
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
    #[case::no_panic(GlobalIndex::new(2, 2, 2), IndexBounds::new(3, 4, 5), RawIndexConversionMode::Border, true, vec![
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
    #[case::no_panic(GlobalIndex::new(2, 2, 2), IndexBounds::new(3, 4, 5), RawIndexConversionMode::Border, false, vec![
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
    #[case::no_panic(LocalIndex::new(2, 2, 2), IndexBounds::new(3, 4, 5), RawIndexConversionMode::Border, false, vec![
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

    #[rstest]
    #[case(IndexBounds::new(2, 2, 3), vec![
        LocalIndex::new(0, 0, 0), LocalIndex::new(1, 0, 0), LocalIndex::new(0, 1, 0), LocalIndex::new(1, 1, 0),
        LocalIndex::new(0, 0, 1), LocalIndex::new(1, 0, 1), LocalIndex::new(0, 1, 1), LocalIndex::new(1, 1, 1),
        LocalIndex::new(0, 0, 2), LocalIndex::new(1, 0, 2), LocalIndex::new(0, 1, 2), LocalIndex::new(1, 1, 2),
    ])]
    fn test_generate_grid_indices(#[case] bounds: IndexBounds, #[case] expected: Vec<LocalIndex>) {
        assert_eq!(generate_grid_indices(&bounds), expected);
    }
}
