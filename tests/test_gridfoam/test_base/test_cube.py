import pytest
import torch

from gridfoam._base._cube import Cube
from gridfoam._base.field_tensor import FieldTensor
from gridfoam.settings import FieldDataAttribute
from gridfoam.utils.enums import CubeType


class TestCube:
    """Test Cube class"""

    @pytest.fixture
    def basic_cube(self) -> Cube:
        """Create a basic Cube for testing"""
        global_index = torch.tensor([123, 456, 789], dtype=torch.int32)
        return Cube(
            width=8,
            bnd_width=2,
            cube_type=CubeType.GHOST_FROM_CHILD,
            depth=5,
            global_index=global_index,
        )

    @pytest.fixture
    def field_data_dict(self) -> dict[str, FieldDataAttribute]:
        """Create field data dictionary for testing"""
        return {
            "U": FieldDataAttribute(shape=(3,), dtype=torch.float32),
            "p": FieldDataAttribute(shape=(1,), dtype=torch.float32),
            "T": FieldDataAttribute(shape=(1,), dtype=torch.float64),
        }

    def test_cube_initialization(self):
        """Test Cube initialization"""
        global_index = torch.tensor([42, 789, 123], dtype=torch.int32)
        face_ids = torch.tensor([1, 4, 7, 9], dtype=torch.int32)
        cube = Cube(
            width=10,
            bnd_width=1,
            cube_type=CubeType.GHOST_FROM_PARENT,
            depth=3,
            global_index=global_index,
            face_ids=face_ids,
            device=torch.device("cuda:0"),
        )

        assert cube.width == 10
        assert cube.bnd_width == 1
        assert cube.cube_type == CubeType.GHOST_FROM_PARENT
        assert cube.depth == 3
        torch.testing.assert_close(cube.global_index, global_index)
        torch.testing.assert_close(cube.face_ids, face_ids)
        assert cube.device == torch.device("cuda:0")
        assert len(cube.field_tensors) == 0

    def test_cube_default_values(self):
        """Test Cube with default values"""
        global_index = torch.tensor([0, 0, 0], dtype=torch.int32)
        face_ids = torch.tensor([], dtype=torch.int32)
        cube = Cube(
            width=8,
            bnd_width=2,
            cube_type=CubeType.LEAF,
            depth=0,
            global_index=global_index,
            face_ids=face_ids,
        )

        assert cube.device == torch.device("cpu")
        assert len(cube.field_tensors) == 0

    def test_allocate_field_tensors(
        self, basic_cube: Cube, field_data_dict: dict[str, FieldDataAttribute]
    ):
        """Test allocate_field_tensors method"""
        basic_cube.allocate_field_tensors(field_data_dict)

        # Check that field tensors were created
        assert len(basic_cube.field_tensors) == 3
        assert "U" in basic_cube.field_tensors
        assert "p" in basic_cube.field_tensors
        assert "T" in basic_cube.field_tensors

        # Check tensor shapes and properties
        u_tensor = basic_cube.field_tensors["U"]
        assert u_tensor.width == 8
        assert u_tensor.bnd == 2
        assert u_tensor.raw.shape == (
            12,
            12,
            12,
            3,
        )  # (width + 2*bnd, width + 2*bnd, width + 2*bnd, channels)
        assert u_tensor.raw.dtype == torch.float32
        assert u_tensor.raw.device == torch.device("cpu")

        p_tensor = basic_cube.field_tensors["p"]
        assert p_tensor.raw.shape == (12, 12, 12, 1)
        assert p_tensor.raw.dtype == torch.float32

        t_tensor = basic_cube.field_tensors["T"]
        assert t_tensor.raw.shape == (12, 12, 12, 1)
        assert t_tensor.raw.dtype == torch.float64

    def test_allocate_field_tensors_different_device(
        self, field_data_dict: dict[str, FieldDataAttribute]
    ):
        """Test allocate_field_tensors with different device"""
        global_index = torch.tensor([1, 1, 1], dtype=torch.int32)
        cube = Cube(
            width=6,
            bnd_width=1,
            cube_type=CubeType.GHOST_FROM_CHILD,
            depth=1,
            global_index=global_index,
            device=torch.device("cuda:0"),
        )

        cube.allocate_field_tensors(field_data_dict)

        # Check that tensors are allocated on the correct device
        for field_tensor in cube.field_tensors.values():
            assert field_tensor.raw.device == torch.device("cuda:0")

    def test_allocate_field_tensors_different_sizes(
        self, field_data_dict: dict[str, FieldDataAttribute]
    ):
        """Test allocate_field_tensors with different cube sizes"""
        global_index = torch.tensor([2, 2, 2], dtype=torch.int32)
        cube = Cube(
            width=4,
            bnd_width=3,
            cube_type=CubeType.LEAF,
            depth=2,
            global_index=global_index,
        )

        cube.allocate_field_tensors(field_data_dict)

        # Check tensor shapes for different sizes
        u_tensor = cube.field_tensors["U"]
        expected_shape = (10, 10, 10, 3)  # (4 + 2*3, 4 + 2*3, 4 + 2*3, 3)
        assert u_tensor.raw.shape == expected_shape

    def test_getitem_access(
        self, basic_cube: Cube, field_data_dict: dict[str, FieldDataAttribute]
    ):
        """Test __getitem__ method for field access"""
        basic_cube.allocate_field_tensors(field_data_dict)

        # Test valid field access
        _ = basic_cube["U"]
        _ = basic_cube["p"]

        # Test invalid field access
        with pytest.raises(AttributeError):
            basic_cube["invalid_field"]

    def test_getattr_access(
        self, basic_cube: Cube, field_data_dict: dict[str, FieldDataAttribute]
    ):
        """Test __getattr__ method for attribute-style access"""
        basic_cube.allocate_field_tensors(field_data_dict)

        # Test valid field access
        _ = basic_cube.U
        _ = basic_cube.p

    def test_items_method(
        self, basic_cube: Cube, field_data_dict: dict[str, FieldDataAttribute]
    ):
        """Test items method"""
        basic_cube.allocate_field_tensors(field_data_dict)

        # Check that values are FieldTensor objects
        for name, field_tensor in basic_cube.items():
            assert name in field_data_dict
            assert isinstance(field_tensor, FieldTensor)

    def test_empty_field_tensors(self, basic_cube: Cube):
        """Test behavior with empty field_tensors"""
        # Test items method with empty field_tensors
        items = list(basic_cube.items())
        assert len(items) == 0

        # Test getitem with empty field_tensors
        with pytest.raises(AttributeError):
            basic_cube["U"]

    def test_cube_types(self):
        """Test different cube types"""
        global_index = torch.tensor([1, 1, 1], dtype=torch.int32)

        # Test LEAF cube
        leaf_cube = Cube(
            width=8,
            bnd_width=2,
            cube_type=CubeType.LEAF,
            depth=1,
            global_index=global_index,
        )
        assert leaf_cube.cube_type == CubeType.LEAF

        # Test GHOST_FROM_PARENT cube
        ghost_parent_cube = Cube(
            width=8,
            bnd_width=2,
            cube_type=CubeType.GHOST_FROM_PARENT,
            depth=1,
            global_index=global_index,
        )
        assert ghost_parent_cube.cube_type == CubeType.GHOST_FROM_PARENT

        # Test GHOST_FROM_CHILD cube
        ghost_child_cube = Cube(
            width=8,
            bnd_width=2,
            cube_type=CubeType.GHOST_FROM_CHILD,
            depth=1,
            global_index=global_index,
        )
        assert ghost_child_cube.cube_type == CubeType.GHOST_FROM_CHILD

    def test_zero_initialization(
        self, basic_cube: Cube, field_data_dict: dict[str, FieldDataAttribute]
    ):
        """Test that allocated tensors are initialized with zeros"""
        basic_cube.allocate_field_tensors(field_data_dict)

        for field_tensor in basic_cube.field_tensors.values():
            # Check that all values are zero
            assert torch.all(field_tensor.raw == 0.0)

    def test_complex_field_shapes(self):
        """Test allocation with complex field shapes"""
        complex_field_data = {
            "vector_field": FieldDataAttribute(shape=(3,), dtype=torch.float32),
            "tensor_field": FieldDataAttribute(
                shape=(3, 3), dtype=torch.float64
            ),
            "scalar_field": FieldDataAttribute(shape=(1,), dtype=torch.float32),
        }

        global_index = torch.tensor([3, 3, 3], dtype=torch.int32)
        cube = Cube(
            width=6,
            bnd_width=1,
            cube_type=CubeType.LEAF,
            depth=3,
            global_index=global_index,
        )

        cube.allocate_field_tensors(complex_field_data)

        # Check shapes for complex fields
        assert cube.field_tensors["vector_field"].raw.shape == (8, 8, 8, 3)
        assert cube.field_tensors["tensor_field"].raw.shape == (8, 8, 8, 3, 3)
        assert cube.field_tensors["scalar_field"].raw.shape == (8, 8, 8, 1)
