import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from gridfoam.utils.enums import AddressMode
from gridfoam.utils.index import (
    generate_grid_indices,
    neighbor_indices,
    ravel_index_3d,
    unravel_index_3d,
)


class TestGetGridIndices:
    """Test cases for generate_grid_indices function."""

    def test_simple_2x2x2_grid(self):
        """Test 2x2x2 grid generation."""
        divisions = torch.tensor([2, 2, 2], dtype=torch.int32)
        indices = generate_grid_indices(divisions)

        expected = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(indices, expected)

    def test_1x1x1_grid(self):
        """Test 1x1x1 grid generation."""
        divisions = torch.tensor([1, 1, 1], dtype=torch.int32)
        indices = generate_grid_indices(divisions)

        expected = torch.tensor([[0, 0, 0]], dtype=torch.int32)
        torch.testing.assert_close(indices, expected)

    def test_3x2x1_grid(self):
        """Test 3x2x1 grid generation."""
        divisions = torch.tensor([3, 2, 1], dtype=torch.int32)
        indices = generate_grid_indices(divisions)

        expected = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]],
            dtype=torch.int32,
        )
        torch.testing.assert_close(indices, expected)


class TestLinearizeGridIndices:
    """Test cases for ravel_index_3d function."""

    def test_simple_2x2x2_grid(self):
        """Test linearization of 2x2x2 grid indices."""
        divisions = torch.tensor([2, 2, 2], dtype=torch.int32)
        indices = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=torch.int32,
        )

        linear_indices = ravel_index_3d(indices, divisions)

        expected = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32)
        torch.testing.assert_close(linear_indices, expected)

    def test_3x2x1_grid(self):
        """Test linearization of 3x2x1 grid indices."""
        divisions = torch.tensor([3, 2, 1], dtype=torch.int32)
        indices = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]],
            dtype=torch.int32,
        )

        linear_indices = ravel_index_3d(indices, divisions)

        expected = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
        torch.testing.assert_close(linear_indices, expected)

    def test_1x1x1_grid(self):
        """Test linearization of 1x1x1 grid indices."""
        divisions = torch.tensor([1, 1, 1], dtype=torch.int32)
        indices = torch.tensor([[0, 0, 0]], dtype=torch.int32)

        linear_indices = ravel_index_3d(indices, divisions)

        expected = torch.tensor([0], dtype=torch.int32)
        torch.testing.assert_close(linear_indices, expected)

    def test_larger_grid(self):
        """Test linearization of a larger grid."""
        divisions = torch.tensor([4, 3, 2], dtype=torch.int32)
        indices = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [3, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [2, 1, 0],
                [3, 1, 0],
                [0, 2, 0],
                [1, 2, 0],
                [2, 2, 0],
                [3, 2, 0],
                [0, 0, 1],
                [1, 0, 1],
                [2, 0, 1],
                [3, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
                [2, 1, 1],
                [3, 1, 1],
                [0, 2, 1],
                [1, 2, 1],
                [2, 2, 1],
                [3, 2, 1],
            ],
            dtype=torch.int32,
        )

        linear_indices = ravel_index_3d(indices, divisions)

        # Verify the formula: index = x + (y + z * div_y) * div_x
        expected = []
        for x, y, z in indices:
            index = x + (y + z * divisions[1]) * divisions[0]
            expected.append(index.item())

        expected = torch.tensor(expected, dtype=torch.int32)
        torch.testing.assert_close(linear_indices, expected)

    def test_round_trip_consistency(self):
        """
        Test that generate_grid_indices and
        ravel_index_3d work together.
        """
        divisions = torch.tensor([3, 2, 2], dtype=torch.int32)

        # Generate grid indices
        indices = generate_grid_indices(divisions)

        # Linearize them
        linear_indices = ravel_index_3d(indices, divisions)

        # Verify we get unique, consecutive indices
        assert linear_indices.shape[0] == 12  # 3 * 2 * 2
        assert set(linear_indices.tolist()) == set(range(12))
        assert linear_indices.tolist() == sorted(linear_indices.tolist())


class TestUnravelIndex3d:
    """Test cases for unravel_index_3d function."""

    def test_unravel_index_3d_single_int(self):
        """Test unravelling of a single integer."""
        divisions = torch.tensor([4, 3, 2], dtype=torch.int32)
        indices = 13
        expected = torch.tensor([1, 0, 1], dtype=torch.int32)
        unraveled_indices = unravel_index_3d(indices, divisions)
        torch.testing.assert_close(unraveled_indices, expected)

    def test_simple_2x2x2_grid(self):
        """Test unravelling of 2x2x2 grid indices."""
        divisions = torch.tensor([2, 2, 2], dtype=torch.int32)
        indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32)

        unraveled_indices = unravel_index_3d(indices, divisions)

        expected = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(unraveled_indices, expected)

    def test_3x2x1_grid(self):
        """Test unravelling of 3x2x1 grid indices."""
        divisions = torch.tensor([3, 2, 1], dtype=torch.int32)
        indices = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
        expected = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]],
            dtype=torch.int32,
        )
        unraveled_indices = unravel_index_3d(indices, divisions)
        torch.testing.assert_close(unraveled_indices, expected)

    def test_1x1x1_grid(self):
        """Test unravelling of 1x1x1 grid indices."""
        divisions = torch.tensor([1, 1, 1], dtype=torch.int32)
        indices = torch.tensor([0], dtype=torch.int32)
        expected = torch.tensor([[0, 0, 0]], dtype=torch.int32)
        unraveled_indices = unravel_index_3d(indices, divisions)
        torch.testing.assert_close(unraveled_indices, expected)

    def test_larger_grid(self):
        """Test unravelling of a larger grid."""
        divisions = torch.tensor([4, 3, 2], dtype=torch.int32)
        indices = torch.arange(24, dtype=torch.int32)
        expected = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [3, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [2, 1, 0],
                [3, 1, 0],
                [0, 2, 0],
                [1, 2, 0],
                [2, 2, 0],
                [3, 2, 0],
                [0, 0, 1],
                [1, 0, 1],
                [2, 0, 1],
                [3, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
                [2, 1, 1],
                [3, 1, 1],
                [0, 2, 1],
                [1, 2, 1],
                [2, 2, 1],
                [3, 2, 1],
            ],
            dtype=torch.int32,
        )
        unraveled_indices = unravel_index_3d(indices, divisions)
        torch.testing.assert_close(unraveled_indices, expected)

    def test_round_trip_consistency(self):
        """
        Test that generate_grid_indices and
        unravel_index_3d work together.
        """
        divisions = torch.tensor([3, 2, 2], dtype=torch.int32)

        # Generate grid indices
        indices = generate_grid_indices(divisions)

        # Linearize them
        linear_indices = ravel_index_3d(indices, divisions)

        # Unravel them
        unraveled_indices = unravel_index_3d(linear_indices, divisions)

        # Verify we get back the original indices
        torch.testing.assert_close(unraveled_indices, indices)


class TestIntegration:
    """Integration tests for both functions."""

    def test_end_to_end_workflow(self):
        """Test complete workflow from divisions to linear indices."""
        divisions = torch.tensor([4, 3, 2], dtype=torch.int32)

        # Step 1: Generate grid indices
        indices = generate_grid_indices(divisions)
        assert indices.shape == (24, 3)  # 4 * 3 * 2 = 24

        # Step 2: Linearize indices
        linear_indices = ravel_index_3d(indices, divisions)
        assert linear_indices.shape[0] == 24

        # Step 3: Verify properties
        assert set(linear_indices.tolist()) == set(range(24))
        assert linear_indices.tolist() == sorted(linear_indices.tolist())

        # Step 4: Verify specific mappings
        # First element should be (0, 0, 0) -> 0
        torch.testing.assert_close(
            indices[0], torch.tensor([0, 0, 0], dtype=torch.int32)
        )
        assert linear_indices[0] == 0

        # Last element should be (3, 2, 1) -> 23
        torch.testing.assert_close(
            indices[-1], torch.tensor([3, 2, 1], dtype=torch.int32)
        )
        assert linear_indices[-1] == 23


class TestNeighborIndices:
    """Test cases for neighbor_indices function."""

    def test_center_point_3x3x3_grid(self):
        """Test neighbor indices for center point in 3x3x3 grid."""
        divisions = torch.tensor([3, 3, 3], dtype=torch.int32)
        indices = torch.tensor([1, 1, 1], dtype=torch.int32)

        neighbors = neighbor_indices(indices, divisions)
        expected = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [2, 1, 0],
                [0, 2, 0],
                [1, 2, 0],
                [2, 2, 0],
                [0, 0, 1],
                [1, 0, 1],
                [2, 0, 1],
                [0, 1, 1],
                [2, 1, 1],
                [0, 2, 1],
                [1, 2, 1],
                [2, 2, 1],
                [0, 0, 2],
                [1, 0, 2],
                [2, 0, 2],
                [0, 1, 2],
                [1, 1, 2],
                [2, 1, 2],
                [0, 2, 2],
                [1, 2, 2],
                [2, 2, 2],
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(neighbors, expected)

    def test_corner_point_2x2x2_grid(self):
        """Test neighbor indices for corner point in 2x2x2 grid."""
        divisions = torch.tensor([2, 2, 2], dtype=torch.int32)
        indices = torch.tensor([0, 0, 0], dtype=torch.int32)

        neighbors = neighbor_indices(indices, divisions)
        expected = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
                [1, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(neighbors, expected)

    def test_include_self(self):
        """Test neighbor indices with include_self=True."""
        divisions = torch.tensor([3, 3, 3], dtype=torch.int32)
        indices = torch.tensor([[1, 1, 1]], dtype=torch.int32)

        neighbors = neighbor_indices(indices, divisions, include_self=True)

        # Should have 27 neighbors (3^3 = 27)
        assert neighbors.shape == (1, 27, 3)

        # Check that center point is included
        center_point = torch.tensor([1, 1, 1], dtype=torch.int32)
        assert (neighbors == center_point).all(dim=-1).any()

    def test_wrap_address_mode(self):
        """Test neighbor indices with WRAP address mode."""
        divisions = torch.tensor([2, 2, 2], dtype=torch.int32)
        indices = torch.tensor([0, 0, 0], dtype=torch.int32)

        neighbors = neighbor_indices(
            indices, divisions, address_mode=AddressMode.WRAP
        )
        expected = torch.tensor(
            [
                [1, 1, 1],
                [0, 1, 1],
                [1, 1, 1],
                [1, 0, 1],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
                [1, 1, 1],
                [1, 1, 0],
                [0, 1, 0],
                [1, 1, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [1, 1, 0],
                [1, 1, 1],
                [0, 1, 1],
                [1, 1, 1],
                [1, 0, 1],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(neighbors, expected)

    def test_border_address_mode(self):
        """Test neighbor indices with BORDER address mode."""
        divisions = torch.tensor([2, 2, 2], dtype=torch.int32)
        indices = torch.tensor([0, 0, 0], dtype=torch.int32)

        neighbors = neighbor_indices(
            indices, divisions, address_mode=AddressMode.BORDER
        )
        expected = torch.tensor(
            [
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [1, 0, 0],
                [-1, -1, -1],
                [0, 1, 0],
                [1, 1, 0],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [0, 0, 1],
                [1, 0, 1],
                [-1, -1, -1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(neighbors, expected)

    def test_multiple_indices(self):
        """Test neighbor indices for multiple input indices."""
        divisions = torch.tensor([3, 3, 3], dtype=torch.int32)
        indices = torch.tensor(
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=torch.int32
        )

        neighbors = neighbor_indices(indices, divisions)

        # Should have shape (3, 26, 3)
        assert neighbors.shape == (3, 26, 3)

        # Each set of neighbors should be valid
        for i in range(3):
            neighbor_set = neighbors[i]
            assert (neighbor_set >= 0).all()
            assert (neighbor_set < divisions).all()

    def test_edge_case_1x1x1_grid(self):
        """Test neighbor indices for 1x1x1 grid."""
        divisions = torch.tensor([1, 1, 1], dtype=torch.int32)
        indices = torch.tensor([[0, 0, 0]], dtype=torch.int32)

        neighbors = neighbor_indices(
            indices, divisions, address_mode=AddressMode.CLAMP
        )

        # Should have 26 neighbors, all clamped to (0,0,0)
        assert neighbors.shape == (1, 26, 3)
        assert (neighbors == 0).all()

    def test_invalid_address_mode(self):
        """Test that invalid address mode raises ValueError."""
        divisions = torch.tensor([2, 2, 2], dtype=torch.int32)
        indices = torch.tensor([[0, 0, 0]], dtype=torch.int32)

        with pytest.raises(BeartypeCallHintParamViolation):
            neighbor_indices(indices, divisions, address_mode="INVALID")
