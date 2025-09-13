use crate::core::constants::{MAX_OCTREE_DEPTH, OCTREE_CODE_BIT_LENGTH, ROOT_CODE_BIT_LENGTH};
use crate::core::errors::MortonError;
use crate::core::types::{
    ChildVector, CubeCode, CubeCodeType, GlobalIndex, GlobalIndexType, LocalIndex, LocalIndexType,
    OctreeCode, OctreeCodeType, RootCode, RootCodeType,
};

use std::hash::{Hash, Hasher};

/// Interleave bits for Morton encoding (3D version)
///
/// This function takes a 32-bit value and interleaves its bits with zeros
/// to create a Morton code. The result can be combined with other axes
/// to create a 3D Morton code.
///
/// # Arguments
///
/// * `x` - The input value to interleave
///
/// # Returns
///
/// A Morton-encoded value with interleaved bits
///
/// # Example
///
/// ```
/// let result = part1by2(0b101);
/// // Result: 0b1001001 (interleaved with zeros)
/// ```
const fn part1by2(x: LocalIndexType) -> OctreeCodeType {
    let mut x = x as OctreeCodeType;
    x = x & 0xFFFFFFFFFFFFFFFF;
    x = (x | (x << 32)) & 0xFFFF00000000FFFF;
    x = (x | (x << 16)) & 0xFF0000FF0000FF0000FF;
    x = (x | (x << 8)) & 0xF00F00F00F00F00F00F00F;
    x = (x | (x << 4)) & 0xC30C30C30C30C30C30C30C3;
    x = (x | (x << 2)) & 0x249249249249249249249249;
    x
}

/// Trait for Morton encoding operations on LocalIndex
///
/// This trait provides methods to encode a 3D local index into Morton codes
/// for octree and root code representations.
pub trait MortonEncodeOps {
    /// Encode the local index into an octree Morton code.
    ///
    /// Returns
    /// -------
    /// OctreeCode
    ///     The Morton-encoded octree code.
    fn to_octree_code(&self) -> OctreeCode;

    /// Encode the local index into an octree Morton code with a specific depth.
    ///
    /// Parameters
    /// ----------
    /// octree_depth : usize
    ///     The depth of the octree to encode for.
    ///
    /// Returns
    /// -------
    /// OctreeCode
    ///     The Morton-encoded octree code, shifted according to the depth.
    fn to_octree_code_with_depth(&self, octree_depth: usize) -> OctreeCode;

    /// Encode the local index into a root code.
    ///
    /// Returns
    /// -------
    /// RootCode
    ///     The Morton-encoded root code.
    fn to_root_code(&self) -> RootCode;
}

impl MortonEncodeOps for LocalIndex {
    fn to_octree_code(&self) -> OctreeCode {
        OctreeCode(part1by2(self[0]) | (part1by2(self[1]) << 1) | (part1by2(self[2]) << 2))
    }

    fn to_octree_code_with_depth(&self, octree_depth: usize) -> OctreeCode {
        let code = self.to_octree_code();
        let shift = 3 * (MAX_OCTREE_DEPTH - octree_depth);
        OctreeCode(code.0 << shift)
    }

    fn to_root_code(&self) -> RootCode {
        RootCode(self.to_octree_code().0 as RootCodeType)
    }
}

/// Extract and compact Morton-encoded bits from a 128-bit value
///
/// This function reverses the Morton encoding process, extracting the original
/// coordinate value from a Morton-encoded 128-bit value.
///
/// # Arguments
///
/// * `octree_code` - The Morton-encoded value to decode
///
/// # Returns
///
/// The original coordinate value before Morton encoding
const fn compact1by2octree(octree_code: OctreeCodeType) -> LocalIndexType {
    let mut x = octree_code & 0x249249249249249249249249;
    x = (x | (x >> 2)) & 0xC30C30C30C30C30C30C30C3;
    x = (x | (x >> 4)) & 0xF00F00F00F00F00F00F00F;
    x = (x | (x >> 8)) & 0xFF0000FF0000FF0000FF;
    x = (x | (x >> 16)) & 0xFFFF00000000FFFF;
    x = (x | (x >> 32)) & 0xFFFFFFFF;
    x as LocalIndexType
}

/// Extract and compact Morton-encoded bits from a 64-bit value
///
/// This function is similar to `compact1by2octree` but works with 64-bit values,
/// used for root code operations.
///
/// # Arguments
///
/// * `root_code` - The Morton-encoded 32-bit value to decode
///
/// # Returns
///
/// The original coordinate value before Morton encoding
const fn compact1by2root(root_code: RootCodeType) -> LocalIndexType {
    let mut x = root_code & 0x09249249;
    x = (x | (x >> 2)) & 0x030C30C3;
    x = (x | (x >> 4)) & 0x0300F00F;
    x = (x | (x >> 8)) & 0x030000FF;
    x as LocalIndexType
}

impl RootCode {
    /// Decode the root code to local index coordinates
    ///
    /// # Returns
    ///
    /// The local index corresponding to this root code
    pub fn decode(&self) -> LocalIndex {
        let x = compact1by2root(self.0);
        let y = compact1by2root(self.0 >> 1);
        let z = compact1by2root(self.0 >> 2);
        LocalIndex::new(x, y, z)
    }
}

impl OctreeCode {
    /// Decode the octree code to local index coordinates
    ///
    /// # Returns
    ///
    /// The local index corresponding to this octree code
    pub fn decode(&self) -> LocalIndex {
        let x = compact1by2octree(self.0);
        let y = compact1by2octree(self.0 >> 1);
        let z = compact1by2octree(self.0 >> 2);
        LocalIndex::new(x, y, z)
    }

    /// Convert the octree code to local index at a specific depth
    ///
    /// # Arguments
    ///
    /// * `depth` - The depth level for the conversion
    ///
    /// # Returns
    ///
    /// The local index at the specified depth
    pub fn to_local_index_with_depth(&self, depth: usize) -> LocalIndex {
        let shift = 3 * (MAX_OCTREE_DEPTH - depth);
        let shift_code = OctreeCode(self.0 >> shift);
        shift_code.decode()
    }

    /// Get the parent octree code at the specified depth
    ///
    /// # Arguments
    ///
    /// * `depth` - The current depth level
    ///
    /// # Returns
    ///
    /// A result containing the parent octree code, or an error if depth is invalid
    pub fn parent(&self, depth: usize) -> Result<OctreeCode, MortonError> {
        if depth == 0 || depth > MAX_OCTREE_DEPTH {
            return Err(MortonError::InvalidDepth { depth });
        }
        let parent_depth = depth - 1;
        let shift = 3 * (MAX_OCTREE_DEPTH - parent_depth);
        let mask = !((1 << shift) - 1);
        Ok(OctreeCode(self.0 & mask))
    }

    /// Get the parent octree code and local offset within the parent
    ///
    /// # Arguments
    ///
    /// * `depth` - The current depth level
    ///
    /// # Returns
    ///
    /// A result containing the parent octree code and local offset, or an error if depth is invalid
    pub fn parent_and_offset(&self, depth: usize) -> Result<(OctreeCode, LocalIndex), MortonError> {
        let parent_code = self.parent(depth)?;
        let local_index = self.to_local_index_with_depth(depth);
        let offsets = LocalIndex::new(local_index[0] & 1, local_index[1] & 1, local_index[2] & 1);
        Ok((parent_code, offsets))
    }

    /// Get the children octree codes at the next depth level
    ///
    /// # Arguments
    ///
    /// * `depth` - The current depth level
    ///
    /// # Returns
    ///
    /// A result containing a vector of 8 child octree codes, or an error if depth is invalid
    pub fn children(&self, depth: usize) -> Result<ChildVector<OctreeCodeType>, MortonError> {
        if depth >= MAX_OCTREE_DEPTH {
            return Err(MortonError::InvalidDepth { depth });
        }
        let child_depth = depth + 1;
        let shift = 3 * (MAX_OCTREE_DEPTH - child_depth);
        let mut result = ChildVector::<OctreeCodeType>::zeros();
        for i in 0..8 {
            result[i] = self.0 + ((i as OctreeCodeType) << shift);
        }
        Ok(result)
    }
}

pub trait ToCubeCode {
    fn to_cubecode(&self, depth: usize) -> CubeCode;
}

impl ToCubeCode for GlobalIndex {
    fn to_cubecode(&self, depth: usize) -> CubeCode {
        let octree_size = 1 << depth;
        let root_index: LocalIndex = self.map(|x| (x / octree_size) as LocalIndexType);
        let octree_index: LocalIndex = self.map(|x| (x % octree_size) as LocalIndexType);
        let root_code = root_index.to_root_code();
        let octree_code = octree_index.to_octree_code_with_depth(depth);
        CubeCode::new(root_code, octree_code)
    }
}

impl CubeCode {
    /// Create a new CubeCode from root code and octree code
    ///
    /// # Arguments
    ///
    /// * `root_code` - The root code for positioning at the root level
    /// * `octree_code` - The octree code for local positioning within the octree
    ///
    /// # Returns
    ///
    /// A new CubeCode combining both codes
    pub fn new(root_code: RootCode, octree_code: OctreeCode) -> Self {
        let root_code = root_code.0 as CubeCodeType;
        let root_code_bit = root_code << OCTREE_CODE_BIT_LENGTH;
        CubeCode(root_code_bit | octree_code.0)
    }

    /// Extract bits from the cube code
    ///
    /// # Arguments
    ///
    /// * `start` - Starting bit position
    /// * `length` - Number of bits to extract
    ///
    /// # Returns
    ///
    /// The extracted bits as a CubeCodeType
    fn extract_bits(&self, start: usize, length: usize) -> CubeCodeType {
        (self.0 >> start) & ((1 << length) - 1)
    }

    /// Parse the cube code into root code and octree code components
    ///
    /// # Returns
    ///
    /// A tuple containing the root code and octree code
    fn parse(&self) -> (RootCode, OctreeCode) {
        let octree_code = self.extract_bits(0, OCTREE_CODE_BIT_LENGTH);
        let root_code = self.extract_bits(OCTREE_CODE_BIT_LENGTH, ROOT_CODE_BIT_LENGTH);
        (RootCode(root_code as RootCodeType), OctreeCode(octree_code))
    }

    /// Get the parent cube code and local offset within the parent
    ///
    /// # Arguments
    ///
    /// * `depth` - The current depth level
    ///
    /// # Returns
    ///
    /// A result containing the parent cube code and local offset, or an error if depth is invalid
    pub fn parent_and_offset(&self, depth: usize) -> Result<(CubeCode, LocalIndex), MortonError> {
        if depth == 0 || depth > MAX_OCTREE_DEPTH {
            return Err(MortonError::InvalidDepth { depth });
        }
        let (root_code, octree_code) = self.parse();
        let (parent_octree_code, offsets) = octree_code.parent_and_offset(depth)?;
        Ok((CubeCode::new(root_code, parent_octree_code), offsets))
    }

    /// Get the children cube codes at the next depth level
    ///
    /// # Arguments
    ///
    /// * `depth` - The current depth level
    ///
    /// # Returns
    ///
    /// A result containing a vector of 8 child cube codes, or an error if depth is invalid
    pub fn children(&self, depth: usize) -> Result<ChildVector<CubeCodeType>, MortonError> {
        if depth >= MAX_OCTREE_DEPTH {
            return Err(MortonError::InvalidDepth { depth });
        }
        let (root_code, octree_code) = self.parse();
        let child_octree_codes = octree_code.children(depth)?;
        Ok(ChildVector::<CubeCodeType>::from_fn(|i, _| {
            CubeCode::new(root_code.clone(), OctreeCode(child_octree_codes[i])).0
        }))
    }

    /// Convert the cube code to a global index at the specified depth
    ///
    /// # Arguments
    ///
    /// * `depth` - The depth level for the conversion
    ///
    /// # Returns
    ///
    /// The global index corresponding to this cube code
    pub fn to_global_index(&self, depth: usize) -> GlobalIndex {
        let octree_size = 1 << depth;
        let (root_code, octree_code) = self.parse();
        let root_index = root_code.decode();
        let local_index = octree_code.to_local_index_with_depth(depth);
        root_index.cast::<GlobalIndexType>() * octree_size + local_index.cast::<GlobalIndexType>()
    }
}

impl Hash for CubeCode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    /// --------------------------------
    /// Octree code
    /// --------------------------------
    #[rstest]
    #[case(LocalIndex::new(1, 4, 7), OctreeCode(0b110100101u128))]
    #[case(LocalIndex::new(2, 5, 8), OctreeCode(0b100010001010u128))]
    #[case(LocalIndex::new(3, 6, 9), OctreeCode(0b100010011101u128))]
    fn test_octree_encode(#[case] input: LocalIndex, #[case] expected: OctreeCode) {
        assert_eq!(input.to_octree_code(), expected);
    }

    #[rstest]
    #[case(
        LocalIndex::new(0b011, 0b101, 0b111),
        MAX_OCTREE_DEPTH,
        OctreeCode(0b110101111u128)
    )]
    #[case(
        LocalIndex::new(2, 3, 1),
        2,
        OctreeCode(0x780000000000000000000000u128)
    )]
    fn test_to_octree_code_with_depth(
        #[case] input: LocalIndex,
        #[case] depth: usize,
        #[case] expected: OctreeCode,
    ) {
        assert_eq!(input.to_octree_code_with_depth(depth), expected);
    }

    #[rstest]
    #[case(OctreeCode(0b110100101u128), LocalIndex::new(1, 4, 7))]
    #[case(OctreeCode(0b100010001010u128), LocalIndex::new(2, 5, 8))]
    #[case(OctreeCode(0b100010011101u128), LocalIndex::new(3, 6, 9))]
    fn test_octree_decode(#[case] input: OctreeCode, #[case] expected: LocalIndex) {
        assert_eq!(input.decode(), expected);
    }

    #[rstest]
    #[case(OctreeCode(0b110101111u128), MAX_OCTREE_DEPTH-1, LocalIndex::new(0b01,0b10, 0b11))]
    #[case(OctreeCode(0b100110101u128), MAX_OCTREE_DEPTH-1, LocalIndex::new(0b00, 0b01, 0b11))]
    #[case(
        OctreeCode(0b110101111u128),
        MAX_OCTREE_DEPTH,
        LocalIndex::new(0b011, 0b101, 0b111)
    )]
    #[case(
        OctreeCode(0b100110101u128),
        MAX_OCTREE_DEPTH,
        LocalIndex::new(0b001, 0b010, 0b111)
    )]
    fn test_octree_to_local_index_with_depth(
        #[case] input: OctreeCode,
        #[case] depth: usize,
        #[case] expected: LocalIndex,
    ) {
        assert_eq!(input.to_local_index_with_depth(depth), expected);
    }

    #[rstest]
    #[case::no_error(
        OctreeCode(0b110101111u128),
        MAX_OCTREE_DEPTH,
        Ok(OctreeCode(0b110101000u128))
    )]
    #[case::invalid_depth(OctreeCode(0), 0, Err(MortonError::InvalidDepth { depth: 0 }))]
    fn test_octree_parent(
        #[case] input: OctreeCode,
        #[case] depth: usize,
        #[case] expected: Result<OctreeCode, MortonError>,
    ) {
        assert_eq!(input.parent(depth), expected);
    }

    #[rstest]
    #[case::no_error(OctreeCode(0b110101111u128), MAX_OCTREE_DEPTH, Ok((OctreeCode(0b110101000u128), LocalIndex::new(1, 1, 1))))]
    #[case::no_error(OctreeCode(0b100110101u128), MAX_OCTREE_DEPTH, Ok((OctreeCode(0b100110000u128), LocalIndex::new(1, 0, 1))))]
    #[case::no_error(OctreeCode(0b100110001u128), MAX_OCTREE_DEPTH, Ok((OctreeCode(0b100110000u128), LocalIndex::new(1, 0, 0))))]
    #[case::invalid_depth(OctreeCode(0), 0, Err(MortonError::InvalidDepth { depth: 0 }))]
    fn test_octree_parent_and_offset(
        #[case] input: OctreeCode,
        #[case] depth: usize,
        #[case] expected: Result<(OctreeCode, LocalIndex), MortonError>,
    ) {
        assert_eq!(input.parent_and_offset(depth), expected);
    }

    #[rstest]
    #[case::no_error(OctreeCode(0b110101000), MAX_OCTREE_DEPTH-1, Ok(ChildVector::<u128>::from([0b110101000,0b110101001,0b110101010,0b110101011,0b110101100,0b110101101,0b110101110,0b110101111])))]
    #[case::invalid_depth(OctreeCode(0b110101111), MAX_OCTREE_DEPTH, Err(MortonError::InvalidDepth { depth: MAX_OCTREE_DEPTH }))]
    fn test_octree_children(
        #[case] input: OctreeCode,
        #[case] depth: usize,
        #[case] expected: Result<ChildVector<u128>, MortonError>,
    ) {
        assert_eq!(input.children(depth), expected);
    }

    /// --------------------------------
    /// Root code
    /// --------------------------------
    #[rstest]
    #[case(LocalIndex::new(1, 4, 7), RootCode(0b110100101u32))]
    #[case(LocalIndex::new(2, 5, 8), RootCode(0b100010001010u32))]
    #[case(LocalIndex::new(3, 6, 9), RootCode(0b100010011101u32))]
    fn test_root_encode(#[case] input: LocalIndex, #[case] expected: RootCode) {
        assert_eq!(input.to_root_code(), expected);
    }

    #[rstest]
    #[case(RootCode(0b110100101u32), LocalIndex::new(1, 4, 7))]
    #[case(RootCode(0b100010001010u32), LocalIndex::new(2, 5, 8))]
    #[case(RootCode(0b100010011101u32), LocalIndex::new(3, 6, 9))]
    fn test_root_decode(#[case] input: RootCode, #[case] expected: LocalIndex) {
        assert_eq!(input.decode(), expected);
    }

    /// --------------------------------
    /// Cubecode
    /// --------------------------------
    #[rstest]
    #[case(GlobalIndex::new(5, 1, 2), 1, CubeCode(0xC600000000000000000000000))]
    #[case(GlobalIndex::new(1, 2, 3), 1, CubeCode(0x6A00000000000000000000000))]
    fn test_cubecode_to_cubecode(
        #[case] global_index: GlobalIndex,
        #[case] depth: usize,
        #[case] expected: CubeCode,
    ) {
        assert_eq!(global_index.to_cubecode(depth), expected);
    }

    #[rstest]
    #[case::no_error(
        CubeCode::new(RootCode(123), OctreeCode(0x2e4a00000000000000000000)),
        5,
        Ok((CubeCode::new(RootCode(123), OctreeCode(0x2E4000000000000000000000)), LocalIndex::new(1, 0, 1)))
    )]
    #[case::no_error(
        CubeCode::new(RootCode(123), OctreeCode(0x600000000000000000000000)),
        1,
        Ok((CubeCode::new(RootCode(123), OctreeCode(0x0)), LocalIndex::new(1, 1, 0)))
    )]
    #[case::invalid_depth(
        CubeCode::new(RootCode(123), OctreeCode(0x600000000000000000000000)),
        0,
        Err(MortonError::InvalidDepth { depth: 0 })
    )]
    #[case::invalid_depth(
        CubeCode::new(RootCode(123), OctreeCode(0x600000000000000000000000)),
        MAX_OCTREE_DEPTH+1,
        Err(MortonError::InvalidDepth { depth: MAX_OCTREE_DEPTH+1 })
    )]
    fn test_cubecode_parent_and_offset(
        #[case] cubecode: CubeCode,
        #[case] depth: usize,
        #[case] expected: Result<(CubeCode, LocalIndex), MortonError>,
    ) {
        assert_eq!(cubecode.parent_and_offset(depth), expected);
    }

    #[rstest]
    #[case::no_error(
        CubeCode::new(RootCode(123), OctreeCode(0x2E4A00000000000000000000)), 5, Ok(ChildVector::<CubeCodeType>::from([
        0x7B2E4A00000000000000000000,
        0x7B2E4A40000000000000000000,
        0x7B2E4A80000000000000000000,
        0x7B2E4AC0000000000000000000,
        0x7B2E4B00000000000000000000,
        0x7B2E4B40000000000000000000,
        0x7B2E4B80000000000000000000,
        0x7B2E4BC0000000000000000000
    ])))]
    #[case::invalid_depth(
        CubeCode::new(RootCode(123), OctreeCode(0x2E4A00000000000000000000)),
        MAX_OCTREE_DEPTH,
        Err(MortonError::InvalidDepth { depth: MAX_OCTREE_DEPTH })
    )]
    fn test_cubecode_children(
        #[case] cubecode: CubeCode,
        #[case] depth: usize,
        #[case] expected: Result<ChildVector<CubeCodeType>, MortonError>,
    ) {
        assert_eq!(cubecode.children(depth), expected);
    }

    #[rstest]
    #[case(CubeCode(0xC600000000000000000000000), 1, GlobalIndex::new(5, 1, 2))]
    #[case(CubeCode(0x6A00000000000000000000000), 1, GlobalIndex::new(1, 2, 3))]
    fn test_cubecode_to_global_index(
        #[case] cubecode: CubeCode,
        #[case] depth: usize,
        #[case] expected: GlobalIndex,
    ) {
        assert_eq!(cubecode.to_global_index(depth), expected);
    }
}
