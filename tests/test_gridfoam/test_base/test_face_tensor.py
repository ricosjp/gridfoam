import pytest
import torch

from gridfoam._base._face_tensor import FaceTensor


class TestFaceTensor:
    """Test FaceTensor class."""

    def test_face_tensor_init_scalar(self) -> None:
        """Test FaceTensor initialization with scalar field."""
        face_tensor = FaceTensor.init(
            w_interior=4,
            shape=(1,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        assert face_tensor.w_interior == 4
        assert face_tensor.x.shape == (1, 4, 4, 5)  # w_interior + 1 for x-faces
        assert face_tensor.y.shape == (1, 4, 5, 4)  # w_interior + 1 for y-faces
        assert face_tensor.z.shape == (1, 5, 4, 4)  # w_interior + 1 for z-faces
        assert face_tensor.x.dtype == torch.float32

    def test_face_tensor_init_vector(self) -> None:
        """Test FaceTensor initialization with vector field."""
        face_tensor = FaceTensor.init(
            w_interior=3,
            shape=(3,),
            dtype=torch.float64,
            device=torch.device("cpu"),
        )

        assert face_tensor.w_interior == 3
        assert face_tensor.x.shape == (3, 3, 3, 4)  # (3, 3, 3, 4)
        assert face_tensor.y.shape == (3, 3, 4, 3)  # (3, 3, 4, 3)
        assert face_tensor.z.shape == (3, 4, 3, 3)  # (3, 4, 3, 3)
        assert face_tensor.x.dtype == torch.float64

    def test_face_tensor_init_tensor_field(self) -> None:
        """Test FaceTensor initialization with tensor field."""
        face_tensor = FaceTensor.init(
            w_interior=2,
            shape=(2, 2),
            dtype=torch.int32,
            device=torch.device("cpu"),
        )

        assert face_tensor.w_interior == 2
        assert face_tensor.x.shape == (2, 2, 2, 2, 3)  # (2, 2, 2, 2, 3)
        assert face_tensor.y.shape == (2, 2, 2, 3, 2)  # (2, 2, 2, 3, 2)
        assert face_tensor.z.shape == (2, 2, 3, 2, 2)  # (2, 2, 3, 2, 2)
        assert face_tensor.x.dtype == torch.int32

    def test_face_tensor_multiplication(self) -> None:
        """Test FaceTensor multiplication."""
        face_tensor1 = FaceTensor.init(
            w_interior=3,
            shape=(1,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        face_tensor2 = FaceTensor.init(
            w_interior=3,
            shape=(1,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Fill with test data
        face_tensor1.x[:] = 2.0
        face_tensor1.y[:] = 3.0
        face_tensor1.z[:] = 4.0

        face_tensor2.x[:] = 5.0
        face_tensor2.y[:] = 6.0
        face_tensor2.z[:] = 7.0

        # Multiply
        result = face_tensor1 * face_tensor2

        # Check results
        torch.testing.assert_close(result.x, torch.ones(1, 3, 3, 4) * 10.0)
        torch.testing.assert_close(result.y, torch.ones(1, 3, 4, 3) * 18.0)
        torch.testing.assert_close(result.z, torch.ones(1, 4, 3, 3) * 28.0)

    def test_get_boundary_face_along_x_axis(self) -> None:
        """Test get_face_along for x-axis."""
        face_tensor = FaceTensor.init(
            w_interior=4,
            shape=(1,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Fill with test data
        face_tensor.x[:] = torch.arange(4 * 4 * 5).float().reshape(1, 4, 4, 5)

        # Test forward face (axis=2, forward=True)
        forward_face = face_tensor.get_boundary_face_along(axis=2, forward=True)
        assert forward_face.shape == (1, 4, 4)

        # Test backward face (axis=2, forward=False)
        backward_face = face_tensor.get_boundary_face_along(
            axis=2, forward=False
        )
        assert backward_face.shape == (1, 4, 4)

    def test_get_boundary_face_along_y_axis(self) -> None:
        """Test get_face_along for y-axis."""
        face_tensor = FaceTensor.init(
            w_interior=3,
            shape=(1,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Fill with test data
        face_tensor.y[:] = torch.arange(3 * 4 * 3).float().reshape(1, 3, 4, 3)

        # Test forward face (axis=1, forward=True)
        forward_face = face_tensor.get_boundary_face_along(axis=1, forward=True)
        assert forward_face.shape == (1, 3, 3)

        # Test backward face (axis=1, forward=False)
        backward_face = face_tensor.get_boundary_face_along(
            axis=1, forward=False
        )
        assert backward_face.shape == (1, 3, 3)

    def test_get_boundary_face_along_z_axis(self) -> None:
        """Test get_face_along for z-axis."""
        face_tensor = FaceTensor.init(
            w_interior=2,
            shape=(1,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Fill with test data
        face_tensor.z[:] = torch.arange(3 * 2 * 2).float().reshape(1, 3, 2, 2)

        # Test forward face (axis=0, forward=True)
        forward_face = face_tensor.get_boundary_face_along(axis=0, forward=True)
        assert forward_face.shape == (1, 2, 2)
        expected_forward = torch.tensor([[[8, 9], [10, 11]]]).float()
        torch.testing.assert_close(forward_face, expected_forward)

        # Test backward face (axis=0, forward=False)
        backward_face = face_tensor.get_boundary_face_along(
            axis=0, forward=False
        )
        assert backward_face.shape == (1, 2, 2)
        expected_backward = torch.tensor([[[0, 1], [2, 3]]]).float()
        torch.testing.assert_close(backward_face, expected_backward)

    def test_get_boundary_face_along_invalid_axis(self) -> None:
        """Test get_face_along with invalid axis."""
        face_tensor = FaceTensor.init(
            w_interior=4,
            shape=(1,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        with pytest.raises(ValueError, match="Invalid axis"):
            face_tensor.get_boundary_face_along(axis=3, forward=True)

    def test_interior_property(self) -> None:
        """Test interior property."""
        face_tensor = FaceTensor.init(
            w_interior=3,
            shape=(1,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Fill with test data
        face_tensor.x[:] = torch.arange(3 * 3 * 4).float().reshape(1, 3, 3, 4)
        face_tensor.y[:] = torch.arange(3 * 4 * 3).float().reshape(1, 3, 4, 3)
        face_tensor.z[:] = torch.arange(4 * 3 * 3).float().reshape(1, 4, 3, 3)

        # Get interior
        interior = face_tensor.interior
        assert interior.shape == (
            1,
            3,
            3,
            3,
            3,
        )  # (3, w_interior, w_interior, w_interior)

        # Check x-component (averaged from x-faces)
        x_comp = interior[0, 0]  # x-component
        expected_x = torch.zeros(3, 3, 3)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    expected_x[i, j, k] = 0.5 * (
                        face_tensor.x[0, i, j, k]
                        + face_tensor.x[0, i, j, k + 1]
                    )
        torch.testing.assert_close(x_comp, expected_x)

    def test_interior_property_vector_field(self) -> None:
        """Test interior property with vector field."""
        face_tensor = FaceTensor.init(
            w_interior=2,
            shape=(3,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Fill with test data
        face_tensor.x[0] = torch.ones(2, 2, 3) * 2.0
        face_tensor.y[1] = torch.ones(2, 3, 2) * 3.0
        face_tensor.z[2] = torch.ones(3, 2, 2) * 4.0

        # Get interior
        interior = face_tensor.interior
        assert interior.shape == (
            3,
            3,
            2,
            2,
            2,
        )  # (3, 3, w_interior, w_interior, w_interior)
        expected_x = torch.zeros(3, 2, 2, 2)
        expected_y = torch.zeros(3, 2, 2, 2)
        expected_z = torch.zeros(3, 2, 2, 2)
        expected_x[0] = 2.0
        expected_y[1] = 3.0
        expected_z[2] = 4.0

        # All components should be the same as input values
        torch.testing.assert_close(interior[0], expected_x)
        torch.testing.assert_close(interior[1], expected_y)
        torch.testing.assert_close(interior[2], expected_z)

    def test_integrate_cell(self) -> None:
        """Test integrate_cell method."""
        face_tensor = FaceTensor.init(
            w_interior=2,
            shape=(1,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Fill with test data
        face_tensor.x[:] = torch.tensor(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
        ).float()

        face_tensor.y[:] = torch.tensor(
            [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
        ).float()

        face_tensor.z[:] = torch.tensor(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]
        ).float()

        # Integrate
        result = face_tensor.integrate_cell()
        assert result.shape == (1, 2, 2, 2)

        # Check first cell (0, 0, 0)
        # x: 2 - 1 = 1, y: 3 - 1 = 2, z: 5 - 1 = 4
        # Total: 1 + 2 + 4 = 7
        assert result[0, 0, 0, 0] == 7.0

    def test_integrate_cell_vector_field(self) -> None:
        """Test integrate_cell with vector field."""
        face_tensor = FaceTensor.init(
            w_interior=2,
            shape=(3,),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Fill with constant values
        face_tensor.x[:] = torch.ones(3, 2, 2, 3) * 2.0
        face_tensor.y[:] = torch.ones(3, 2, 3, 2) * 3.0
        face_tensor.z[:] = torch.ones(3, 3, 2, 2) * 4.0

        # Integrate
        result = face_tensor.integrate_cell()
        assert result.shape == (3, 2, 2, 2)

        # All components should be zero (constant field)
        torch.testing.assert_close(result, torch.zeros(3, 2, 2, 2))
