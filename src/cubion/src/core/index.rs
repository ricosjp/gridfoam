use crate::core::errors::IndexError;
use crate::core::types::{
    CubeCodeType, GlobalIndex, GlobalIndexType, IndexBounds, LocalIndex, LocalIndexType,
    OctreeCodeType, RawIndex, RawIndexConversionMode, RawIndexType,
};

pub trait RawIndexExt {
    fn convert_to<T>(self, mode: RawIndexConversionMode, bounds: &IndexBounds) -> Option<T>
    where
        T: CastRawIndex;
}

impl RawIndexExt for RawIndex {
    fn convert_to<T>(self, mode: RawIndexConversionMode, bounds: &IndexBounds) -> Option<T>
    where
        T: CastRawIndex,
    {
        match mode {
            RawIndexConversionMode::Wrap => {
                let bounds = bounds.cast::<RawIndexType>();
                let wrapped = self.zip_map(&bounds, |i, b| (i % b + b) % b);
                Some(T::try_from_raw_index(&wrapped).unwrap())
            }
            RawIndexConversionMode::Clamp => {
                let bounds = bounds.cast::<RawIndexType>();
                let clipped = self.zip_map(&bounds, |i, b| i.clamp(0, b - 1));
                Some(T::try_from_raw_index(&clipped).unwrap())
            }
            RawIndexConversionMode::Mirror => {
                let bounds = bounds.cast::<RawIndexType>();
                let mirrored = self.zip_map(&bounds, |i, b| {
                    let p = 2 * b;
                    let r = ((i % p) + p) % p;
                    b - 1 - (r - b + 1).abs()
                });
                Some(T::try_from_raw_index(&mirrored).unwrap())
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

        // Use try_cast to convert to GlobalIndexType
        raw_index
            .try_cast::<GlobalIndexType>()
            .ok_or(IndexError::IndexOverflow {
                raw_index: *raw_index,
            })
    }

    fn to_raw_index(&self) -> RawIndex {
        self.cast::<RawIndexType>()
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

        // Use try_cast to convert to LocalIndexType
        raw_index
            .try_cast::<LocalIndexType>()
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
            && self.x < bounds.x as RawIndexType
            && self.y >= 0
            && self.y < bounds.y as RawIndexType
            && self.z >= 0
            && self.z < bounds.z as RawIndexType
    }
}

impl IsInBounds for GlobalIndex {
    fn is_in_bounds(&self, bounds: &IndexBounds) -> bool {
        self.x < bounds.x && self.y < bounds.y && self.z < bounds.z
    }
}

impl IsInBounds for LocalIndex {
    fn is_in_bounds(&self, bounds: &IndexBounds) -> bool {
        self.x < bounds.x as LocalIndexType
            && self.y < bounds.y as LocalIndexType
            && self.z < bounds.z as LocalIndexType
    }
}

/// Trait for converting multi-dimensional coordinates to flat indices
///
/// This trait provides methods to convert 3D grid coordinates to 1D array indices,
/// which is useful for storing grid data in flat arrays.
pub trait RavelMultiIndex {
    type Output;
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
    fn ravel_multi_index(&self, bounds: &IndexBounds) -> Self::Output;
}

impl RavelMultiIndex for GlobalIndex {
    type Output = CubeCodeType;
    fn ravel_multi_index(&self, bounds: &IndexBounds) -> Self::Output {
        let index = self.cast::<CubeCodeType>();
        let bounds = bounds.cast::<CubeCodeType>();
        index.x + index.y * bounds.x + index.z * bounds.x * bounds.y
    }
}

impl RavelMultiIndex for LocalIndex {
    type Output = OctreeCodeType;
    fn ravel_multi_index(&self, bounds: &IndexBounds) -> Self::Output {
        let index = self.cast::<OctreeCodeType>();
        let bounds = bounds.cast::<OctreeCodeType>();
        index.x + index.y * bounds.x + index.z * bounds.x * bounds.y
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

impl UnravelIndex for OctreeCodeType {
    type Output = Result<LocalIndex, IndexError>;
    fn unravel_index(self, bounds: &IndexBounds) -> Self::Output {
        let bounds_u128 = bounds.cast::<OctreeCodeType>();
        let total_size = bounds_u128.x * bounds_u128.y * bounds_u128.z;
        if self >= total_size {
            return Err(IndexError::RaveledIndexOutOfBounds {
                raveled_index: self,
                bounds: *bounds,
            });
        }

        let x = self % bounds_u128.x;
        let y = (self / bounds_u128.x) % bounds_u128.y;
        let z = self / (bounds_u128.x * bounds_u128.y);
        Ok(LocalIndex::new(
            u32::try_from(x).unwrap(),
            u32::try_from(y).unwrap(),
            u32::try_from(z).unwrap(),
        ))
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
                result.push(LocalIndex::new(
                    x as LocalIndexType,
                    y as LocalIndexType,
                    z as LocalIndexType,
                ));
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::wrap(
        RawIndex::new(3, 1, 4),
        RawIndexConversionMode::Wrap,
        IndexBounds::new(6, 7, 4),
        Some(LocalIndex::new(3, 1, 0))
    )]
    #[case::clamp(
        RawIndex::new(3, 1, 4),
        RawIndexConversionMode::Clamp,
        IndexBounds::new(6, 7, 4),
        Some(LocalIndex::new(3, 1, 3))
    )]
    #[case::mirror(
        RawIndex::new(3, 1, 4),
        RawIndexConversionMode::Mirror,
        IndexBounds::new(6, 7, 4),
        Some(LocalIndex::new(3, 1, 2))
    )]
    #[case::border(
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
        assert_eq!(input.convert_to::<LocalIndex>(mode, &bounds), expected);
    }

    #[rstest]
    #[case::wrap(
        RawIndex::new(3, 1, 4),
        RawIndexConversionMode::Wrap,
        IndexBounds::new(6, 7, 4),
        Some(GlobalIndex::new(3, 1, 0))
    )]
    #[case::clamp(
        RawIndex::new(3, 1, 4),
        RawIndexConversionMode::Clamp,
        IndexBounds::new(6, 7, 4),
        Some(GlobalIndex::new(3, 1, 3))
    )]
    #[case::mirror(
        RawIndex::new(3, 1, 4),
        RawIndexConversionMode::Mirror,
        IndexBounds::new(6, 7, 4),
        Some(GlobalIndex::new(3, 1, 2))
    )]
    #[case::border(
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
        assert_eq!(input.convert_to::<GlobalIndex>(mode, &bounds), expected);
    }

    #[rstest]
    #[case(LocalIndex::new(0, 1, 2), IndexBounds::new(1, 2, 3), 0+1*1+2*1*2)]
    #[case(LocalIndex::new(3, 1, 2), IndexBounds::new(6, 7, 4), 3+1*6+2*6*7)]
    fn test_ravel_multi_index_local(
        #[case] input: LocalIndex,
        #[case] bounds: IndexBounds,
        #[case] expected: OctreeCodeType,
    ) {
        assert_eq!(input.ravel_multi_index(&bounds), expected);
    }

    #[rstest]
    #[case(GlobalIndex::new(0, 1, 2), IndexBounds::new(1, 2, 3), 0+1*1+2*1*2)]
    #[case(GlobalIndex::new(3, 1, 2), IndexBounds::new(6, 7, 4), 3+1*6+2*6*7)]
    fn test_ravel_multi_index_global(
        #[case] input: GlobalIndex,
        #[case] bounds: IndexBounds,
        #[case] expected: OctreeCodeType,
    ) {
        assert_eq!(input.ravel_multi_index(&bounds), expected);
    }

    #[rstest]
    #[case(12, IndexBounds::new(3, 4, 5), Ok(LocalIndex::new(0, 0, 1)))]
    #[case(27, IndexBounds::new(3, 4, 5), Ok(LocalIndex::new(0, 1, 2)))]
    #[case(43, IndexBounds::new(3, 4, 5), Ok(LocalIndex::new(1, 2, 3)))]
    #[case::out_of_bounds(75, IndexBounds::new(3, 4, 5), Err(IndexError::RaveledIndexOutOfBounds { raveled_index: 75, bounds: IndexBounds::new(3, 4, 5) }))]
    fn test_unravel_index(
        #[case] input: OctreeCodeType,
        #[case] bounds: IndexBounds,
        #[case] expected: Result<LocalIndex, IndexError>,
    ) {
        assert_eq!(input.unravel_index(&bounds), expected);
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
