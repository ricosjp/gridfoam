import pytest
import torch

from gridfoam.utils.cube_code import (
    _extract_bits,
    child_cube_codes,
    code_to_global_index,
    gen_cube_code,
    global_indices_to_codes,
    parent_cube_code_and_offsets,
    parse_cube_code,
)
from gridfoam.utils.enums import Constants


class TestExtractBits:
    """Test cases for _extract_bits function."""

    def test_extract_bits_simple(self):
        """Test simple bit extraction."""
        # Test value: 0b1010101010101010
        test_value = 0xAAAA
        result = _extract_bits(test_value, 4, 8)
        expected = 0xAA  # 0b10101010
        assert result == expected

    def test_extract_bits_from_start(self):
        """Test bit extraction from the start."""
        # Test value: 0b1111000011110000
        test_value = 0xF0F0
        result = _extract_bits(test_value, 0, 4)
        expected = 0x0  # 0b0000
        assert result == expected

    def test_extract_bits_from_middle(self):
        """Test bit extraction from the middle."""
        # Test value: 0b1111000011110000
        test_value = 0xF0F0
        result = _extract_bits(test_value, 4, 8)
        expected = 0xF  # 0b00001111
        assert result == expected

    def test_extract_bits_single_bit(self):
        """Test extraction of a single bit."""
        # Test value: 0b1010101010101010
        test_value = 0xAAAA
        result = _extract_bits(test_value, 7, 1)
        expected = 1  # 0b1
        assert result == expected

    def test_extract_bits_invalid_start(self):
        """Test that negative start raises ValueError."""
        with pytest.raises(ValueError, match="start must be >= 0"):
            _extract_bits(0xFFFF, -1, 4)

    def test_extract_bits_invalid_length(self):
        """Test that non-positive length raises ValueError."""
        with pytest.raises(ValueError, match="length must be > 0"):
            _extract_bits(0xFFFF, 5, 0)


class TestGenCubeCode:
    """Test cases for gen_cube_code function."""

    def test_gen_cube_code_simple(self):
        """Test simple cube code generation."""
        root_code = 123
        morton_code = 456
        expected_code = 2268949521066274849224
        cube_code = gen_cube_code(root_code, morton_code)
        assert cube_code == expected_code

    def test_gen_cube_code_zero_values(self):
        """Test cube code generation with zero values."""
        root_code = 0
        morton_code = 0

        cube_code = gen_cube_code(root_code, morton_code)
        assert cube_code == 0

    def test_gen_cube_code_max_values(self):
        """Test cube code generation with maximum values."""
        root_code = (1 << Constants.ROOT_CODE_BIT_LENGTH) - 1
        morton_code = (1 << Constants.MORTON_CODE_BIT_LENGTH) - 1
        expected_code = 0xFFFFFFFFFFFFFFFFFFFFF
        cube_code = gen_cube_code(root_code, morton_code)
        assert cube_code == expected_code


class TestParseCubeCode:
    """Test cases for parse_cube_code function."""

    def test_parse_cube_code_simple(self):
        """Test simple cube code parsing."""
        root_code = 123
        morton_code = 456

        cube_code = gen_cube_code(root_code, morton_code)
        parsed_root, parsed_morton = parse_cube_code(cube_code)

        assert parsed_root == root_code
        assert parsed_morton == morton_code

    def test_parse_cube_code_zero(self):
        """Test parsing of zero cube code."""
        cube_code = 0
        root_code, morton_code = parse_cube_code(cube_code)

        assert root_code == 0
        assert morton_code == 0

    def test_parse_cube_code_max(self):
        """Test parsing of maximum cube code."""
        max_root = (1 << Constants.ROOT_CODE_BIT_LENGTH) - 1
        max_morton = (1 << Constants.MORTON_CODE_BIT_LENGTH) - 1

        cube_code = gen_cube_code(max_root, max_morton)
        root_code, morton_code = parse_cube_code(cube_code)

        assert root_code == max_root
        assert morton_code == max_morton

    def test_parse_cube_code_round_trip(self):
        """Test round trip: generate -> parse -> generate."""
        original_root = 42
        original_morton = 12345

        cube_code = gen_cube_code(original_root, original_morton)
        parsed_root, parsed_morton = parse_cube_code(cube_code)
        regenerated_code = gen_cube_code(parsed_root, parsed_morton)

        assert regenerated_code == cube_code


class TestChildCubeCodes:
    """Test cases for child_cube_codes function."""

    def test_child_cube_codes_normal_depth(self):
        """Test child cube code generation for normal depth."""
        depth = 5
        root_code = 123
        morton_code = 0x2E4A00000000000

        cube_code = gen_cube_code(root_code, morton_code)
        child_codes = child_cube_codes(cube_code, depth)

        expected_child_mortons = [
            0x2E4A00000000000,
            0x2E4A40000000000,
            0x2E4A80000000000,
            0x2E4AC0000000000,
            0x2E4B00000000000,
            0x2E4B40000000000,
            0x2E4B80000000000,
            0x2E4BC0000000000,
        ]

        for child_code, expected_child_morton in zip(
            child_codes, expected_child_mortons, strict=False
        ):
            expected_child_code = gen_cube_code(
                root_code, expected_child_morton
            )
            assert child_code == expected_child_code

    def test_child_cube_codes_max_depth(self):
        """Test child cube code generation at maximum depth."""
        depth = Constants.MAX_OCTREE_DEPTH
        root_code = 123
        morton_code = 0xFFFFFFFFFFFFFFF

        cube_code = gen_cube_code(root_code, morton_code)
        with pytest.raises(
            ValueError,
            match=f"Depth must be less than {Constants.MAX_OCTREE_DEPTH}",
        ):
            child_cube_codes(cube_code, depth)

    def test_child_cube_codes_zero_depth(self):
        """Test child cube code generation at zero depth."""
        depth = 0
        root_code = 123
        morton_code = 0

        cube_code = gen_cube_code(root_code, morton_code)
        child_codes = child_cube_codes(cube_code, depth)

        expected_child_mortons = [
            0x0,
            0x200000000000000,
            0x400000000000000,
            0x600000000000000,
            0x800000000000000,
            0xA00000000000000,
            0xC00000000000000,
            0xE00000000000000,
        ]

        for child_code, expected_child_morton in zip(
            child_codes, expected_child_mortons, strict=False
        ):
            expected_child_code = gen_cube_code(
                root_code, expected_child_morton
            )
            assert child_code == expected_child_code


class TestParentCubeCodeAndOffsets:
    """Test cases for parent_cube_code_and_offsets function."""

    def test_parent_cube_code_and_offsets_normal_depth(self):
        """Test parent cube code generation for normal depth."""
        depth = 5
        root_code = 123
        morton_code = 0x2E4A00000000000

        cube_code = gen_cube_code(root_code, morton_code)
        parent_code, offsets = parent_cube_code_and_offsets(cube_code, depth)

        expected_parent_morton = 0x2E4000000000000
        expected_parent_code = gen_cube_code(root_code, expected_parent_morton)

        assert parent_code == expected_parent_code
        assert offsets == (1, 0, 1)

    def test_parent_cube_code_and_offsets_zero_depth(self):
        """Test parent cube code generation at zero depth."""
        depth = 0
        root_code = 123
        morton_code = 0

        cube_code = gen_cube_code(root_code, morton_code)
        with pytest.raises(ValueError, match="Depth must be greater than 0"):
            parent_cube_code_and_offsets(cube_code, depth)

    def test_parent_cube_code_and_offsets_depth_one(self):
        """Test parent cube code generation at depth one."""
        depth = 1
        root_code = 123
        morton_code = 0x600000000000000

        cube_code = gen_cube_code(root_code, morton_code)
        parent_code, offsets = parent_cube_code_and_offsets(cube_code, depth)

        expected_parent_morton = 0x0
        expected_parent_code = gen_cube_code(root_code, expected_parent_morton)

        assert parent_code == expected_parent_code
        assert offsets == (1, 1, 0)

    def test_parent_child_round_trip(self):
        """Test round trip: child -> parent -> child."""
        depth = 5
        root_code = 123
        morton_code = 0x2E4A00000000000

        cube_code = gen_cube_code(root_code, morton_code)
        parent_code, offsets = parent_cube_code_and_offsets(cube_code, depth)

        parent_children = child_cube_codes(parent_code, depth - 1)
        offset = offsets[0] + offsets[1] * 2 + offsets[2] * 4
        assert parent_children[offset] == cube_code


class TestGlobalIndicesToCodes:
    """Test cases for global_indices_to_codes function."""

    def test_global_indices_to_codes_simple(self):
        """Test simple global index to code conversion."""
        global_indices = torch.tensor([[5, 1, 2], [1, 2, 3]], dtype=torch.int32)
        block_divisions = torch.tensor([4, 2, 2], dtype=torch.int32)
        depth = 1
        expected_codes = [0xA0600000000000000, 0xC0A00000000000000]
        cube_codes = global_indices_to_codes(
            global_indices, block_divisions, depth
        )
        for cube_code, expected_code in zip(
            cube_codes, expected_codes, strict=True
        ):
            assert cube_code == expected_code

    def test_global_indices_to_codes_round_trip(self):
        """
        Test round trip:
        global indices to codes -> codes to global indices.
        """
        n_indices = 20
        depth = 5
        block_divisions = torch.tensor([4, 2, 2], dtype=torch.int32)
        colx = torch.randint(
            0,
            block_divisions[0] * (1 << depth),
            (n_indices,),
            dtype=torch.int32,
        )
        coly = torch.randint(
            0,
            block_divisions[1] * (1 << depth),
            (n_indices,),
            dtype=torch.int32,
        )
        colz = torch.randint(
            0,
            block_divisions[2] * (1 << depth),
            (n_indices,),
            dtype=torch.int32,
        )
        global_indices = torch.stack([colx, coly, colz], dim=1)
        cube_codes = global_indices_to_codes(
            global_indices, block_divisions, depth
        )
        result = []
        for cube_code in cube_codes:
            global_indices_2 = code_to_global_index(
                cube_code, block_divisions, depth
            )
            result.append(global_indices_2)
        torch.testing.assert_close(global_indices, torch.stack(result, dim=0))


class TestCodeToGlobalIndex:
    """Test cases for code_to_global_index function."""

    def test_code_to_global_index_simple(self):
        """Test simple code to global index conversion."""
        cube_codes = [0xA0600000000000000, 0xC0A00000000000000]
        block_divisions = torch.tensor([4, 2, 2], dtype=torch.int32)
        depth = 1
        expected_global_indices = torch.tensor(
            [[5, 1, 2], [1, 2, 3]], dtype=torch.int32
        )
        for cube_code, expected_global_index in zip(
            cube_codes, expected_global_indices, strict=True
        ):
            global_index = code_to_global_index(
                cube_code, block_divisions, depth
            )
            torch.testing.assert_close(global_index, expected_global_index)
