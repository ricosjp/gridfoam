import torch

from gridfoam._base._field import Field


class TestField:
    """Test Field class."""

    def test_field_init(self) -> None:
        """Test Field initialization."""
        field = Field(
            w_interior=8,
            w_halo=2,
            device=torch.device("cpu"),
        )

        assert field.w_interior == 8
        assert field.w_halo == 2
        assert field.device == torch.device("cpu")
        assert field.cells == {}
        assert field.faces == {}

    def test_add_cell_tensor_scalar(self) -> None:
        """Test adding a scalar cell tensor."""
        field = Field(
            w_interior=4,
            w_halo=1,
            device=torch.device("cpu"),
        )

        field.add_cell_tensor("pressure", (1,), torch.float32)

        assert "pressure" in field.cells
        assert "pressure" not in field.faces
        cell_tensor = field.cells["pressure"]
        assert cell_tensor.w_interior == 4
        assert cell_tensor.w_halo == 1
        assert cell_tensor.raw.shape == (1, 6, 6, 6)  # 4 + 2*1
        assert cell_tensor.raw.dtype == torch.float32

    def test_add_cell_tensor_vector(self) -> None:
        """Test adding a vector cell tensor."""
        field = Field(
            w_interior=6,
            w_halo=3,
            device=torch.device("cpu"),
        )

        field.add_cell_tensor("velocity", (3,), torch.float64)

        assert "velocity" in field.cells
        cell_tensor = field.cells["velocity"]
        assert cell_tensor.w_interior == 6
        assert cell_tensor.w_halo == 3
        assert cell_tensor.raw.shape == (3, 12, 12, 12)  # 6 + 2*3
        assert cell_tensor.raw.dtype == torch.float64

    def test_add_cell_tensor_tensor_field(self) -> None:
        """Test adding a tensor field cell tensor."""
        field = Field(
            w_interior=2,
            w_halo=1,
            device=torch.device("cpu"),
        )

        field.add_cell_tensor("stress", (2, 2), torch.int32)

        assert "stress" in field.cells
        cell_tensor = field.cells["stress"]
        assert cell_tensor.w_interior == 2
        assert cell_tensor.w_halo == 1
        assert cell_tensor.raw.shape == (2, 2, 4, 4, 4)  # 2 + 2*1
        assert cell_tensor.raw.dtype == torch.int32

    def test_add_face_tensor_scalar(self) -> None:
        """Test adding a scalar face tensor."""
        field = Field(
            w_interior=4,
            w_halo=1,
            device=torch.device("cpu"),
        )

        field.add_face_tensor("flux", (), torch.float32)

        assert "flux" in field.faces
        assert "flux" not in field.cells
        face_tensor = field.faces["flux"]
        assert face_tensor.w_interior == 4
        assert face_tensor.x.shape == (4, 4, 5)  # w_interior + 1 for x-faces
        assert face_tensor.y.shape == (4, 5, 4)  # w_interior + 1 for y-faces
        assert face_tensor.z.shape == (5, 4, 4)  # w_interior + 1 for z-faces
        assert face_tensor.x.dtype == torch.float32

    def test_add_face_tensor_vector(self) -> None:
        """Test adding a vector face tensor."""
        field = Field(
            w_interior=3,
            w_halo=2,
            device=torch.device("cpu"),
        )

        field.add_face_tensor("momentum", (3,), torch.float64)

        assert "momentum" in field.faces
        face_tensor = field.faces["momentum"]
        assert face_tensor.w_interior == 3
        assert face_tensor.x.shape == (3, 3, 3, 4)  # (3, 3, 3, 4)
        assert face_tensor.y.shape == (3, 3, 4, 3)  # (3, 3, 4, 3)
        assert face_tensor.z.shape == (3, 4, 3, 3)  # (3, 4, 3, 3)
        assert face_tensor.x.dtype == torch.float64

    def test_add_face_tensor_tensor_field(self) -> None:
        """Test adding a tensor field face tensor."""
        field = Field(
            w_interior=2,
            w_halo=1,
            device=torch.device("cpu"),
        )

        field.add_face_tensor("gradient", (2, 2), torch.int32)

        assert "gradient" in field.faces
        face_tensor = field.faces["gradient"]
        assert face_tensor.w_interior == 2
        assert face_tensor.x.shape == (2, 2, 2, 2, 3)  # (2, 2, 2, 2, 3)
        assert face_tensor.y.shape == (2, 2, 2, 3, 2)  # (2, 2, 2, 3, 2)
        assert face_tensor.z.shape == (2, 2, 3, 2, 2)  # (2, 2, 3, 2, 2)
        assert face_tensor.x.dtype == torch.int32
