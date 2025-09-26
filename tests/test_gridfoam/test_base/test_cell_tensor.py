import pytest
import torch

from gridfoam._base._cell_tensor import CellTensor, grad


class TestCellTensor:
    """Test CellTensor class."""

    def test_cell_tensor_init_scalar(self) -> None:
        """Test CellTensor initialization with scalar field."""
        cell_tensor = CellTensor.init(
            w_interior=8,
            w_halo=2,
            shape=(1,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        assert cell_tensor.w_interior == 8
        assert cell_tensor.w_halo == 2
        assert cell_tensor.ndim == 1
        assert cell_tensor.raw.shape == (1, 12, 12, 12)  # 8 + 2*2
        assert cell_tensor.raw.dtype == torch.float32

    def test_cell_tensor_init_vector(self) -> None:
        """Test CellTensor initialization with vector field."""
        cell_tensor = CellTensor.init(
            w_interior=4,
            w_halo=1,
            shape=(3,),
            dtype=torch.float64,
            device=torch.device("cpu"),
        )

        assert cell_tensor.w_interior == 4
        assert cell_tensor.w_halo == 1
        assert cell_tensor.ndim == 1
        assert cell_tensor.raw.shape == (3, 6, 6, 6)  # 4 + 2*1
        assert cell_tensor.raw.dtype == torch.float64

    def test_cell_tensor_init_tensor_field(self) -> None:
        """Test CellTensor initialization with tensor field."""
        cell_tensor = CellTensor.init(
            w_interior=6,
            w_halo=3,
            shape=(2, 2),
            dtype=torch.int32,
            device=torch.device("cpu"),
        )

        assert cell_tensor.w_interior == 6
        assert cell_tensor.w_halo == 3
        assert cell_tensor.ndim == 2
        assert cell_tensor.raw.shape == (2, 2, 12, 12, 12)  # 6 + 2*3
        assert cell_tensor.raw.dtype == torch.int32

    def test_cell_tensor_init_invalid_shape(self) -> None:
        """Test CellTensor initialization with invalid shape."""
        with pytest.raises(ValueError, match="Invalid shape"):
            CellTensor.init(
                w_interior=8,
                w_halo=2,
                shape=(1, 2, 3),  # 3D field not allowed
                dtype=torch.float32,
                device=torch.device("cpu"),
            )

    def test_interior_shape(self) -> None:
        """Test interior_shape property."""
        cell_tensor = CellTensor.init(
            w_interior=8,
            w_halo=2,
            shape=(3,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        expected_shape = (3, 8, 8, 8)
        assert cell_tensor.interior_shape == expected_shape

    def test_interior_property(self) -> None:
        """Test interior property getter and setter."""
        cell_tensor = CellTensor.init(
            w_interior=4,
            w_halo=1,
            shape=(1,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Test setter
        test_data = torch.ones(1, 4, 4, 4)
        cell_tensor.interior = test_data
        torch.testing.assert_close(cell_tensor.interior, test_data)

    def test_get_half_interior(self) -> None:
        """Test get_half_interior method."""
        cell_tensor = CellTensor.init(
            w_interior=8,
            w_halo=2,
            shape=(1,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Fill with test data
        cell_tensor.raw[:] = torch.arange(12**3).float().reshape(1, 12, 12, 12)

        # Test offset [0, 0, 0] - left half
        offset = torch.tensor([0, 0, 0], dtype=torch.int32)
        half_interior = cell_tensor.get_half_interior(offset)
        assert half_interior.shape == (1, 4, 4, 4)

        # Test offset [1, 1, 1] - right half
        offset = torch.tensor([1, 1, 1], dtype=torch.int32)
        half_interior = cell_tensor.get_half_interior(offset)
        assert half_interior.shape == (1, 4, 4, 4)

    def test_get_halo_along(self) -> None:
        """Test get_halo_along method."""
        cell_tensor = CellTensor.init(
            w_interior=4,
            w_halo=2,
            shape=(1,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Fill with test data
        cell_tensor.raw[:] = torch.arange(8**3).float().reshape(1, 8, 8, 8)

        # Test forward halo along x-axis (axis=2)
        halo = cell_tensor.get_halo_along(axis=2, forward=True)
        assert halo.shape == (1, 4, 4, 2)  # interior x interior x halo

        # Test backward halo along y-axis (axis=1)
        halo = cell_tensor.get_halo_along(axis=1, forward=False)
        assert halo.shape == (1, 4, 2, 4)  # interior x halo x interior

    def test_set_halo_along(self) -> None:
        """Test set_halo_along method."""
        cell_tensor = CellTensor.init(
            w_interior=4,
            w_halo=2,
            shape=(1,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Create test data for halo
        halo_data = torch.ones(1, 2, 4, 4) * 42.0

        # Set forward halo along z-axis (axis=0)
        cell_tensor.set_halo_along(axis=0, forward=True, value=halo_data)

        # Verify the halo was set correctly
        retrieved_halo = cell_tensor.get_halo_along(axis=0, forward=True)
        torch.testing.assert_close(retrieved_halo, halo_data)

    def test_get_interior_halo_along(self) -> None:
        """Test get_interior_halo_along method."""
        cell_tensor = CellTensor.init(
            w_interior=4,
            w_halo=2,
            shape=(1,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Fill with test data
        cell_tensor.raw[:] = torch.arange(8**3).float().reshape(1, 8, 8, 8)

        # Test forward interior halo along x-axis (axis=2)
        interior_halo = cell_tensor.get_interior_halo_along(
            axis=2, forward=True
        )
        assert interior_halo.shape == (1, 4, 4, 2)  # interior x interior x halo

        # Test backward interior halo along y-axis (axis=1)
        interior_halo = cell_tensor.get_interior_halo_along(
            axis=1, forward=False
        )
        assert interior_halo.shape == (1, 4, 2, 4)  # interior x halo x interior

    def test_get_shifted_interior_along(self) -> None:
        """Test get_shifted_interior_along method."""
        cell_tensor = CellTensor.init(
            w_interior=4,
            w_halo=2,
            shape=(1,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Fill with test data
        cell_tensor.raw[:] = torch.arange(8**3).float().reshape(1, 8, 8, 8)

        # Test forward shifted interior along x-axis (axis=2)
        shifted = cell_tensor.get_shifted_interior_along(axis=2, forward=True)
        assert shifted.shape == (1, 4, 4, 4)  # same as interior

        # Test backward shifted interior along y-axis (axis=1)
        shifted = cell_tensor.get_shifted_interior_along(axis=1, forward=False)
        assert shifted.shape == (1, 4, 4, 4)  # same as interior

    def test_face_average(self) -> None:
        """Test face_average with a linear gradient."""
        cell_tensor = CellTensor.init(
            w_interior=4,
            w_halo=2,
            shape=(1,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Create a linear gradient in x-direction
        for i in range(8):
            cell_tensor.raw[0, :, :, i] = float(i)

        # Compute face average
        face_tensor = cell_tensor.face_average()

        # x-face values should be averages of adjacent cells
        expected_x = torch.tensor(
            [
                [
                    [
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                    ],
                    [
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                    ],
                    [
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                    ],
                    [
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                        [1.5, 2.5, 3.5, 4.5, 5.5],
                    ],
                ]
            ]
        )

        torch.testing.assert_close(face_tensor.x, expected_x)


class TestGrad:
    """Test grad function."""

    def test_grad_constant_field(self) -> None:
        """Test gradient of a constant field."""
        cell_tensor = CellTensor.init(
            w_interior=4,
            w_halo=2,
            shape=(1,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Fill with constant value
        cell_tensor.raw[:] = 5.0

        # Compute gradient
        dx = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        grad_tensor = grad(cell_tensor, dx)

        # Gradient of constant field should be zero
        torch.testing.assert_close(grad_tensor.x, torch.zeros(1, 4, 4, 5))
        torch.testing.assert_close(grad_tensor.y, torch.zeros(1, 4, 5, 4))
        torch.testing.assert_close(grad_tensor.z, torch.zeros(1, 5, 4, 4))

    def test_grad_linear_field(self) -> None:
        """Test gradient of a linear field."""
        cell_tensor = CellTensor.init(
            w_interior=4,
            w_halo=2,
            shape=(1,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Create linear field: f(x,y,z) = 2x + 3y + 4z
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    cell_tensor.raw[0, i, j, k] = 2.0 * k + 3.0 * j + 4.0 * i

        # Compute gradient
        dx = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        grad_tensor = grad(cell_tensor, dx)

        # Expected gradients: df/dx = 2, df/dy = 3, df/dz = 4
        expected_x = torch.ones(1, 4, 4, 5) * 2.0
        expected_y = torch.ones(1, 4, 5, 4) * 3.0
        expected_z = torch.ones(1, 5, 4, 4) * 4.0

        torch.testing.assert_close(grad_tensor.x, expected_x)
        torch.testing.assert_close(grad_tensor.y, expected_y)
        torch.testing.assert_close(grad_tensor.z, expected_z)

    def test_grad_different_spacing(self) -> None:
        """Test gradient with different grid spacing."""
        cell_tensor = CellTensor.init(
            w_interior=4,
            w_halo=2,
            shape=(1,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Create linear field: f(x,y,z) = x + y + z
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    cell_tensor.raw[0, i, j, k] = float(k + j + i)

        # Compute gradient with different spacing
        dx = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)
        grad_tensor = grad(cell_tensor, dx)

        # Expected gradients: df/dx = 1/2, df/dy = 1/3, df/dz = 1/4
        expected_x = torch.ones(1, 4, 4, 5) * 0.5
        expected_y = torch.ones(1, 4, 5, 4) * (1.0 / 3.0)
        expected_z = torch.ones(1, 5, 4, 4) * 0.25

        torch.testing.assert_close(grad_tensor.x, expected_x)
        torch.testing.assert_close(grad_tensor.y, expected_y)
        torch.testing.assert_close(grad_tensor.z, expected_z)

    def test_grad_vector_field(self) -> None:
        """Test gradient of a vector field."""
        cell_tensor = CellTensor.init(
            w_interior=4,
            w_halo=2,
            shape=(3,),  # 3D vector field
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Create vector field with linear components
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    cell_tensor.raw[0, i, j, k] = float(k)  # x-component
                    cell_tensor.raw[1, i, j, k] = float(j)  # y-component
                    cell_tensor.raw[2, i, j, k] = float(i)  # z-component

        # Compute gradient
        dx = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        grad_tensor = grad(cell_tensor, dx)

        # Expected gradients for each component
        # dfx/dx = 1, dfx/dy = 0, dfx/dz = 0
        # dfy/dx = 0, dfy/dy = 1, dfy/dz = 0
        # dfz/dx = 0, dfz/dy = 0, dfz/dz = 1

        expected_x = torch.zeros(3, 4, 4, 5)
        expected_x[0] = 1.0  # dfx/dx = 1

        expected_y = torch.zeros(3, 4, 5, 4)
        expected_y[1] = 1.0  # dfy/dy = 1

        expected_z = torch.zeros(3, 5, 4, 4)
        expected_z[2] = 1.0  # dfz/dz = 1

        torch.testing.assert_close(grad_tensor.x, expected_x)
        torch.testing.assert_close(grad_tensor.y, expected_y)
        torch.testing.assert_close(grad_tensor.z, expected_z)
