import torch

from gridfoam.utils.flag import dilate_sparse_coords


class TestDilateSparseCoords:
    """Test cases for dilate_sparse_coords function."""

    def test_single_coordinate_center(self):
        """Test dilation of a single coordinate at the center of the domain."""
        flag_coords = torch.tensor([[5, 5, 5]], dtype=torch.int32)
        bounds = torch.tensor([10, 10, 10], dtype=torch.int32)

        result = dilate_sparse_coords(flag_coords, bounds)

        # Should have 27 coordinates (3x3x3 neighborhood)
        assert result.shape == (27, 3)
        assert torch.all(result >= 0)
        assert torch.all(result[:, 0] < 10)
        assert torch.all(result[:, 1] < 10)
        assert torch.all(result[:, 2] < 10)

    def test_single_coordinate_corner(self):
        """Test dilation of a single coordinate at the corner of the domain."""
        flag_coords = torch.tensor([[0, 0, 0]], dtype=torch.int32)
        bounds = torch.tensor([5, 5, 5], dtype=torch.int32)

        result = dilate_sparse_coords(flag_coords, bounds)

        # Should have 8 coordinates (only positive offsets are valid)
        assert result.shape == (8, 3)
        assert torch.all(result >= 0)
        assert torch.all(result[:, 0] < 5)
        assert torch.all(result[:, 1] < 5)
        assert torch.all(result[:, 2] < 5)

    def test_single_coordinate_edge(self):
        """Test dilation of a single coordinate at the edge of the domain."""
        flag_coords = torch.tensor([[0, 5, 5]], dtype=torch.int32)
        bounds = torch.tensor([10, 10, 10], dtype=torch.int32)

        result = dilate_sparse_coords(flag_coords, bounds)

        # Should have 18 coordinates (2x3x3 neighborhood)
        assert result.shape == (18, 3)
        assert torch.all(result >= 0)
        assert torch.all(result[:, 0] < 10)
        assert torch.all(result[:, 1] < 10)
        assert torch.all(result[:, 2] < 10)

    def test_multiple_coordinates(self):
        """Test dilation of multiple coordinates."""
        flag_coords = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.int32)
        bounds = torch.tensor([5, 5, 5], dtype=torch.int32)

        result = dilate_sparse_coords(flag_coords, bounds)

        # Should have fewer than 54(=27*2) coordinates due to overlap
        assert result.shape[0] <= 54
        assert result.shape[1] == 3
        assert torch.all(result >= 0)
        assert torch.all(result[:, 0] < 5)
        assert torch.all(result[:, 1] < 5)
        assert torch.all(result[:, 2] < 5)

    def test_adjacent_coordinates(self):
        """Test dilation of adjacent coordinates that should overlap."""
        flag_coords = torch.tensor([[1, 1, 1], [1, 1, 2]], dtype=torch.int32)
        bounds = torch.tensor([5, 5, 5], dtype=torch.int32)

        result = dilate_sparse_coords(flag_coords, bounds)

        # Should have fewer than 54(=27*2) coordinates due to overlap
        assert result.shape[0] < 54
        assert result.shape[1] == 3

    def test_empty_input(self):
        """Test with empty input coordinates."""
        flag_coords = torch.empty((0, 3), dtype=torch.int32)
        bounds = torch.tensor([5, 5, 5], dtype=torch.int32)

        result = dilate_sparse_coords(flag_coords, bounds)

        assert result.shape == (0, 3)

    def test_small_domain(self):
        """Test with very small domain bounds."""
        flag_coords = torch.tensor([[0, 0, 0]], dtype=torch.int32)
        bounds = torch.tensor([1, 1, 1], dtype=torch.int32)

        result = dilate_sparse_coords(flag_coords, bounds)

        # Should have only 1 coordinate (the original)
        assert result.shape == (1, 3)
        assert torch.all(result == flag_coords)

    def test_rectangular_domain(self):
        """Test with rectangular domain (different dimensions)."""
        flag_coords = torch.tensor([[1, 2, 3]], dtype=torch.int32)
        bounds = torch.tensor([3, 5, 7], dtype=torch.int32)

        result = dilate_sparse_coords(flag_coords, bounds)

        # Should have 27 coordinates initially, but some may be filtered
        assert result.shape[0] <= 27
        assert result.shape[1] == 3
        assert torch.all(result[:, 0] < 3)
        assert torch.all(result[:, 1] < 5)
        assert torch.all(result[:, 2] < 7)

    def test_coordinates_at_boundary(self):
        """Test coordinates exactly at the boundary."""
        flag_coords = torch.tensor([[4, 4, 4]], dtype=torch.int32)
        bounds = torch.tensor([5, 5, 5], dtype=torch.int32)

        result = dilate_sparse_coords(flag_coords, bounds)

        # Should have 8 coordinates (only negative offsets are valid)
        assert result.shape == (8, 3)
        assert torch.all(result >= 0)
        assert torch.all(result[:, 0] < 5)
        assert torch.all(result[:, 1] < 5)
        assert torch.all(result[:, 2] < 5)

    def test_no_duplicates(self):
        """Test that the result contains no duplicate coordinates."""
        flag_coords = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.int32)
        bounds = torch.tensor([5, 5, 5], dtype=torch.int32)

        result = dilate_sparse_coords(flag_coords, bounds)

        # Should have 27 unique coordinates (not 54)
        assert result.shape == (27, 3)

        # Check for duplicates
        unique_coords = torch.unique(result, dim=0)
        assert unique_coords.shape == result.shape
