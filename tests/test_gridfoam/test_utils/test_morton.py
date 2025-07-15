import pytest
import torch

from gridfoam.utils.enums import Constants
from gridfoam.utils.morton import (
    get_ancestor_code,
    get_child_codes,
    get_local_index,
    get_parent_code,
    morton_decode,
    morton_encode,
    part1by2,
    unpart1by2,
)


class TestPart1By2:
    """Test part1by2 function"""

    @pytest.mark.parametrize(
        "n, expected_code",
        [
            (0, torch.tensor([0b0], dtype=torch.int64)),
            (1, torch.tensor([0b1], dtype=torch.int64)),
            (2, torch.tensor([0b1000], dtype=torch.int64)),
            (3, torch.tensor([0b1001], dtype=torch.int64)),
            (7, torch.tensor([0b1001001], dtype=torch.int64)),
        ],
    )
    def test_part1by2(self, n: int, expected_code: int):
        """Test part1by2 with a single value"""
        n = torch.tensor([n], dtype=torch.int32)
        result = part1by2(n)
        torch.testing.assert_close(result, expected_code)

    @pytest.mark.parametrize(
        "n, expected_code",
        [
            (
                torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32),
                torch.tensor(
                    [
                        0b0,
                        0b1,
                        0b1000,
                        0b1001,
                        0b1000000,
                        0b1000001,
                        0b1001000,
                        0b1001001,
                    ],
                    dtype=torch.int64,
                ),
            ),
        ],
    )
    def test_part1by2_multiple_values(
        self, n: torch.Tensor, expected_code: torch.Tensor
    ):
        """Test part1by2 with multiple values"""
        result = part1by2(n)
        torch.testing.assert_close(result, expected_code)

    def test_part1by2_large_value(self):
        """Test part1by2 with large values"""
        n = torch.tensor([0x1FFFFF], dtype=torch.int32)  # Max 21-bit value
        result = part1by2(n)
        expected_code = torch.tensor(
            [0b1001001001001001001001001001001001001001001001001001001001001],
            dtype=torch.int64,
        )
        torch.testing.assert_close(result, expected_code)


class TestUnpart1By2:
    """Test unpart1by2 function"""

    @pytest.mark.parametrize(
        "n, expected_code",
        [
            (
                torch.tensor([0b0], dtype=torch.int64),
                torch.tensor([0], dtype=torch.int32),
            ),
            (
                torch.tensor([0b1], dtype=torch.int64),
                torch.tensor([1], dtype=torch.int32),
            ),
            (
                torch.tensor([0b1000], dtype=torch.int64),
                torch.tensor([2], dtype=torch.int32),
            ),
            (
                torch.tensor([0b1001], dtype=torch.int64),
                torch.tensor([3], dtype=torch.int32),
            ),
            (
                torch.tensor([0b1001001], dtype=torch.int64),
                torch.tensor([7], dtype=torch.int32),
            ),
        ],
    )
    def test_unpart1by2_single_value(self, n: torch.Tensor, expected_code: int):
        """Test unpart1by2 with a single value"""
        result = unpart1by2(n)
        torch.testing.assert_close(result, expected_code)

    @pytest.mark.parametrize(
        "n, expected_code",
        [
            (
                torch.tensor(
                    [
                        0b0,
                        0b1,
                        0b1000,
                        0b1001,
                        0b1000000,
                        0b1000001,
                        0b1001000,
                        0b1001001,
                    ],
                    dtype=torch.int64,
                ),
                torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32),
            ),
        ],
    )
    def test_unpart1by2_multiple_values(
        self, n: torch.Tensor, expected_code: torch.Tensor
    ):
        """Test unpart1by2 with multiple values"""
        result = unpart1by2(n)
        torch.testing.assert_close(result, expected_code)

    def test_unpart1by2_large_value(self):
        """Test part1by2 with large values"""
        n = torch.tensor(
            [0b1001001001001001001001001001001001001001001001001001001001001],
            dtype=torch.int64,
        )  # Max 21-bit value
        result = unpart1by2(n)
        expected_code = torch.tensor(
            [0x1FFFFF],
            dtype=torch.int32,
        )
        torch.testing.assert_close(result, expected_code)


class TestMortonEncode:
    """Test morton_encode function"""

    def test_morton_encode_single_point(self):
        """Test morton_encode with single point"""
        ix = torch.tensor([1], dtype=torch.int32)
        iy = torch.tensor([2], dtype=torch.int32)
        iz = torch.tensor([3], dtype=torch.int32)

        result = morton_encode(ix, iy, iz)
        expected_code = torch.tensor([0b110101], dtype=torch.int64)
        torch.testing.assert_close(result, expected_code)

    def test_morton_encode_multiple_points(self):
        """Test morton_encode with multiple points"""
        ix = torch.tensor([1, 2, 3], dtype=torch.int32)
        iy = torch.tensor([4, 5, 6], dtype=torch.int32)
        iz = torch.tensor([7, 8, 9], dtype=torch.int32)

        result = morton_encode(ix, iy, iz)
        expected_code = torch.tensor(
            [0b110100101, 0b100010001010, 0b100010011101], dtype=torch.int64
        )
        torch.testing.assert_close(result, expected_code)

    def test_morton_encode_zero_coordinates(self):
        """Test morton_encode with zero coordinates"""
        ix = torch.tensor([0], dtype=torch.int32)
        iy = torch.tensor([0], dtype=torch.int32)
        iz = torch.tensor([0], dtype=torch.int32)

        result = morton_encode(ix, iy, iz)
        assert result.dtype == torch.int64
        assert result[0] == 0

    def test_morton_encode_large_coordinates(self):
        """Test morton_encode with large coordinates"""
        ix = torch.tensor([0x1FFFFF], dtype=torch.int32)  # Max 21-bit
        iy = torch.tensor([0x1FFFFF], dtype=torch.int32)
        iz = torch.tensor([0x1FFFFF], dtype=torch.int32)

        result = morton_encode(ix, iy, iz)
        expected_code = torch.tensor(
            [0b111111111111111111111111111111111111111111111111111111111111111],
            dtype=torch.int64,
        )
        torch.testing.assert_close(result, expected_code)


class TestMortonDecode:
    """Test morton_decode function"""

    def test_morton_decode_single_code(self):
        """Test morton_decode with single code"""
        code = torch.tensor([0b110101], dtype=torch.int64)
        ix, iy, iz = morton_decode(code)

        torch.testing.assert_close(ix, torch.tensor([1], dtype=torch.int32))
        torch.testing.assert_close(iy, torch.tensor([2], dtype=torch.int32))
        torch.testing.assert_close(iz, torch.tensor([3], dtype=torch.int32))

    def test_morton_decode_multiple_codes(self):
        """Test morton_decode with multiple codes"""
        code = torch.tensor(
            [0b110100101, 0b100010001010, 0b100010011101], dtype=torch.int64
        )
        ix, iy, iz = morton_decode(code)

        torch.testing.assert_close(
            ix, torch.tensor([1, 2, 3], dtype=torch.int32)
        )
        torch.testing.assert_close(
            iy, torch.tensor([4, 5, 6], dtype=torch.int32)
        )
        torch.testing.assert_close(
            iz, torch.tensor([7, 8, 9], dtype=torch.int32)
        )

    def test_morton_decode_zero_code(self):
        """Test morton_decode with zero code"""
        code = torch.tensor([0], dtype=torch.int64)
        ix, iy, iz = morton_decode(code)

        assert ix[0] == 0
        assert iy[0] == 0
        assert iz[0] == 0


class TestMortonEncodeDecodeRoundTrip:
    """Test round-trip encoding and decoding"""

    def test_round_trip_single_point(self):
        """Test encode -> decode round trip with single point"""
        ix = torch.tensor([42], dtype=torch.int32)
        iy = torch.tensor([17], dtype=torch.int32)
        iz = torch.tensor([99], dtype=torch.int32)

        code = morton_encode(ix, iy, iz)
        decoded_ix, decoded_iy, decoded_iz = morton_decode(code)

        assert torch.all(decoded_ix == ix)
        assert torch.all(decoded_iy == iy)
        assert torch.all(decoded_iz == iz)

    def test_round_trip_multiple_points(self):
        """Test encode -> decode round trip with multiple points"""
        ix = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
        iy = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32)
        iz = torch.tensor([100, 200, 300, 400, 500], dtype=torch.int32)

        code = morton_encode(ix, iy, iz)
        decoded_ix, decoded_iy, decoded_iz = morton_decode(code)

        assert torch.all(decoded_ix == ix)
        assert torch.all(decoded_iy == iy)
        assert torch.all(decoded_iz == iz)

    def test_round_trip_large_values(self):
        """Test encode -> decode round trip with large values"""
        ix = torch.tensor([0x1FFFFF], dtype=torch.int32)
        iy = torch.tensor([0x1FFFFF], dtype=torch.int32)
        iz = torch.tensor([0x1FFFFF], dtype=torch.int32)

        code = morton_encode(ix, iy, iz)
        decoded_ix, decoded_iy, decoded_iz = morton_decode(code)

        assert torch.all(decoded_ix == ix)
        assert torch.all(decoded_iy == iy)
        assert torch.all(decoded_iz == iz)


class TestGetAncestorCode:
    """Test get_ancestor_code function"""

    def test_get_ancestor_code_same_depth(self):
        """Test get_ancestor_code when depths are the same"""
        code = 12345
        current_depth = 5
        ancestor_depth = 5

        result = get_ancestor_code(code, current_depth, ancestor_depth)
        assert result == code

    def test_get_ancestor_code_parent(self):
        """Test get_ancestor_code for parent"""
        code = 0b110101111
        current_depth = 20
        ancestor_depth = 19

        result = get_ancestor_code(code, current_depth, ancestor_depth)
        expected_code = 0b110101000
        assert result == expected_code

    def test_get_ancestor_code_grandparent(self):
        """Test get_ancestor_code for grandparent"""
        code = 0b110101111
        current_depth = 20
        ancestor_depth = 18

        result = get_ancestor_code(code, current_depth, ancestor_depth)
        expected_code = 0b110000000
        assert result == expected_code

    def test_get_ancestor_code_invalid_depth(self):
        """Test get_ancestor_code with invalid depth"""
        code = 12345
        current_depth = 5
        ancestor_depth = 6

        with pytest.raises(ValueError):
            get_ancestor_code(code, current_depth, ancestor_depth)


class TestGetParentCode:
    """Test get_parent_code function"""

    def test_get_parent_code(self):
        """Test get_parent_code"""
        code = 0b110101111
        octree_depth = 20

        result = get_parent_code(code, octree_depth)
        expected_code = 0b110101000
        assert result == expected_code

    def test_get_parent_code_depth_zero(self):
        """Test get_parent_code with depth 0"""
        code = 12345
        octree_depth = 0

        with pytest.raises(ValueError):
            get_parent_code(code, octree_depth)


class TestGetChildCodes:
    """Test get_child_codes function"""

    def test_get_child_codes(self):
        """Test get_child_codes"""
        code = 0b110101000
        octree_depth = 19

        child_codes = torch.tensor(
            get_child_codes(code, octree_depth), dtype=torch.int64
        )
        expected_codes = torch.tensor(
            [
                0b110101000,
                0b110101001,
                0b110101010,
                0b110101011,
                0b110101100,
                0b110101101,
                0b110101110,
                0b110101111,
            ],
            dtype=torch.int64,
        )
        torch.testing.assert_close(child_codes, expected_codes)

    def test_get_child_codes_max_depth(self):
        """Test get_child_codes at max depth"""
        code = 0b110101111
        octree_depth = Constants.MAX_OCTREE_DEPTH

        with pytest.raises(ValueError):
            get_child_codes(code, octree_depth)

    def test_get_child_codes_above_max_depth(self):
        """Test get_child_codes above max depth"""
        code = 0b110101111
        octree_depth = Constants.MAX_OCTREE_DEPTH + 1

        with pytest.raises(ValueError):
            get_child_codes(code, octree_depth)


class TestGetLocalIndex:
    """Test get_local_index function"""

    def test_get_local_index(self):
        """Test get_local_index"""
        code = 0b110101111
        octree_depth = 19

        result = get_local_index(code, octree_depth)
        expected_result = torch.tensor([0b01, 0b10, 0b11], dtype=torch.int32)
        torch.testing.assert_close(result, expected_result)

    def test_get_local_index_max_depth(self):
        """Test get_local_index at max depth"""
        code = 0b110101111
        octree_depth = Constants.MAX_OCTREE_DEPTH

        result = get_local_index(code, octree_depth)
        expected_result = torch.tensor([0b011, 0b101, 0b111], dtype=torch.int32)
        torch.testing.assert_close(result, expected_result)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_part1by2_unpart1by2_round_trip(self):
        """Test part1by2 -> unpart1by2 round trip"""
        n = torch.tensor([42], dtype=torch.int32)
        encoded = part1by2(n)
        decoded = unpart1by2(encoded)
        torch.testing.assert_close(decoded, n)

    def test_morton_operations_with_negative_values(self):
        """Test that negative values are handled appropriately"""
        n = torch.tensor([-1], dtype=torch.int32)
        result = part1by2(n)
        expected_code = torch.tensor(
            [0b1001001001001001001001001001001001001001001001001001001001001],
            dtype=torch.int64,
        )
        torch.testing.assert_close(result, expected_code)
