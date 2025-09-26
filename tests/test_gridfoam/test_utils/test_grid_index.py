import torch

from gridfoam.utils.grid_index import generate_grid_indices


class TestGenerateGridIndices:
    """Test generate_grid_indices function."""

    def test_generate_grid_indices_basic(self) -> None:
        """Test basic functionality of generate_grid_indices."""
        divisions = torch.tensor([2, 2, 2], dtype=torch.int32)
        indices = generate_grid_indices(divisions)

        # Should have 2*2*2 = 8 indices
        assert indices.shape == (8, 3)
        assert indices.dtype == torch.int32

        # Check that all indices are within bounds
        assert torch.all(indices >= 0)
        assert torch.all(indices[:, 0] < 2)  # z
        assert torch.all(indices[:, 1] < 2)  # y
        assert torch.all(indices[:, 2] < 2)  # x

    def test_generate_grid_indices_single_division(self) -> None:
        """Test with single division along each axis."""
        divisions = torch.tensor([1, 1, 1], dtype=torch.int32)
        indices = generate_grid_indices(divisions)

        # Should have 1*1*1 = 1 index
        assert indices.shape == (1, 3)
        expected = torch.tensor([[0, 0, 0]], dtype=torch.int32)
        torch.testing.assert_close(indices, expected)

    def test_generate_grid_indices_different_sizes(self) -> None:
        """Test with different sizes along each axis."""
        divisions = torch.tensor([3, 2, 4], dtype=torch.int32)
        indices = generate_grid_indices(divisions)

        # Should have 3*2*4 = 24 indices
        assert indices.shape == (24, 3)

        # Check bounds
        assert torch.all(indices[:, 0] < 4)  # z
        assert torch.all(indices[:, 1] < 2)  # y
        assert torch.all(indices[:, 2] < 3)  # x

        # Check that all combinations are present
        unique_indices = torch.unique(indices, dim=0)
        assert unique_indices.shape[0] == 24

    def test_generate_grid_indices_z_order(self) -> None:
        """Test that indices are ordered in Z-order."""
        divisions = torch.tensor([2, 2, 2], dtype=torch.int32)
        indices = generate_grid_indices(divisions)

        # Expected Z-order for 2x2x2 grid:
        # (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
        expected = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=torch.int32,
        )

        torch.testing.assert_close(indices, expected)

    def test_generate_grid_indices_edge_case_zero(self) -> None:
        """Test edge case with one dimension being zero."""
        divisions = torch.tensor([2, 0, 3], dtype=torch.int32)
        indices = generate_grid_indices(divisions)

        # Should have 2*0*3 = 0 indices
        assert indices.shape == (0, 3)
