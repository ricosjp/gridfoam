import torch

from gridfoam.utils.index import get_grid_indices, linearize_grid_indices


class TestGetGridIndices:
    """Test cases for get_grid_indices function."""

    def test_simple_2x2x2_grid(self):
        """Test 2x2x2 grid generation."""
        divisions = torch.tensor([2, 2, 2], dtype=torch.int32)
        indices = get_grid_indices(divisions)

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
        indices = get_grid_indices(divisions)

        expected = torch.tensor([[0, 0, 0]], dtype=torch.int32)
        torch.testing.assert_close(indices, expected)

    def test_3x2x1_grid(self):
        """Test 3x2x1 grid generation."""
        divisions = torch.tensor([3, 2, 1], dtype=torch.int32)
        indices = get_grid_indices(divisions)

        expected = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]],
            dtype=torch.int32,
        )
        torch.testing.assert_close(indices, expected)


class TestLinearizeGridIndices:
    """Test cases for linearize_grid_indices function."""

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

        linear_indices = linearize_grid_indices(indices, divisions)

        expected = [0, 1, 2, 3, 4, 5, 6, 7]
        assert linear_indices == expected

    def test_3x2x1_grid(self):
        """Test linearization of 3x2x1 grid indices."""
        divisions = torch.tensor([3, 2, 1], dtype=torch.int32)
        indices = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]],
            dtype=torch.int32,
        )

        linear_indices = linearize_grid_indices(indices, divisions)

        expected = [0, 1, 2, 3, 4, 5]
        assert linear_indices == expected

    def test_1x1x1_grid(self):
        """Test linearization of 1x1x1 grid indices."""
        divisions = torch.tensor([1, 1, 1], dtype=torch.int32)
        indices = torch.tensor([[0, 0, 0]], dtype=torch.int32)

        linear_indices = linearize_grid_indices(indices, divisions)

        expected = [0]
        assert linear_indices == expected

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

        linear_indices = linearize_grid_indices(indices, divisions)

        # Verify the formula: index = x + (y + z * div_y) * div_x
        expected = []
        for x, y, z in indices:
            index = x + (y + z * divisions[1]) * divisions[0]
            expected.append(index.item())

        assert linear_indices == expected

    def test_round_trip_consistency(self):
        """Test that get_grid_indices and linearize_grid_indices work together."""
        divisions = torch.tensor([3, 2, 2], dtype=torch.int32)

        # Generate grid indices
        indices = get_grid_indices(divisions)

        # Linearize them
        linear_indices = linearize_grid_indices(indices, divisions)

        # Verify we get unique, consecutive indices
        assert len(linear_indices) == 12  # 3 * 2 * 2
        assert set(linear_indices) == set(range(12))
        assert linear_indices == sorted(linear_indices)


class TestIntegration:
    """Integration tests for both functions."""

    def test_end_to_end_workflow(self):
        """Test complete workflow from divisions to linear indices."""
        divisions = torch.tensor([4, 3, 2], dtype=torch.int32)

        # Step 1: Generate grid indices
        indices = get_grid_indices(divisions)
        assert indices.shape == (24, 3)  # 4 * 3 * 2 = 24

        # Step 2: Linearize indices
        linear_indices = linearize_grid_indices(indices, divisions)
        assert len(linear_indices) == 24

        # Step 3: Verify properties
        assert set(linear_indices) == set(range(24))
        assert linear_indices == sorted(linear_indices)

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
