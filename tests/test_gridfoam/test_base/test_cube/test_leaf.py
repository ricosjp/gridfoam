import pytest
import torch

from gridfoam._base.cube import LeafCube
from gridfoam._base.field_tensor import FieldTensor
from gridfoam.settings import FieldDataAttribute
from gridfoam.utils.cube_code import gen_cube_code
from gridfoam.utils.enums import CubeType


class TestLeafCube:
    """Test LeafCube class"""

    @pytest.fixture
    def basic_leaf_cube(self) -> LeafCube:
        """Create a basic LeafCube for testing"""
        cube_code = gen_cube_code(root_code=123, morton_code=456)
        face_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        neighbor_codes = [
            gen_cube_code(root_code=123, morton_code=457),
            gen_cube_code(root_code=123, morton_code=458),
            gen_cube_code(root_code=124, morton_code=456),
        ]
        return LeafCube(
            width=8,
            bnd_width=2,
            depth=5,
            cube_code=cube_code,
            face_ids=face_ids,
            neighbor_codes=neighbor_codes,
        )

    @pytest.fixture
    def field_data_dict(self) -> dict[str, FieldDataAttribute]:
        """Create field data dictionary for testing"""
        return {
            "U": FieldDataAttribute(shape=(3,), dtype=torch.float32),
            "p": FieldDataAttribute(shape=(1,), dtype=torch.float32),
        }

    def test_leaf_cube_initialization(self):
        """Test LeafCube initialization"""
        cube_code = gen_cube_code(root_code=42, morton_code=789)
        face_ids = torch.tensor([10, 20, 30], dtype=torch.int32)
        neighbor_codes = [
            gen_cube_code(root_code=42, morton_code=790),
            gen_cube_code(root_code=43, morton_code=789),
        ]

        leaf_cube = LeafCube(
            width=10,
            bnd_width=1,
            depth=3,
            cube_code=cube_code,
            face_ids=face_ids,
            neighbor_codes=neighbor_codes,
            device=torch.device("cuda:0"),
        )

        assert leaf_cube.width == 10
        assert leaf_cube.bnd_width == 1
        assert leaf_cube.cube_type == CubeType.LEAF
        assert leaf_cube.cube_code == cube_code
        assert leaf_cube.device == torch.device("cuda:0")
        assert torch.equal(leaf_cube.face_ids, face_ids)
        assert leaf_cube.neighbor_codes == neighbor_codes
        assert len(leaf_cube.field_tensors) == 0

    def test_allocate_field_tensors_inheritance(
        self,
        basic_leaf_cube: LeafCube,
        field_data_dict: dict[str, FieldDataAttribute],
    ):
        """Test that LeafCube inherits allocate_field_tensors from Cube"""
        basic_leaf_cube.allocate_field_tensors(field_data_dict)

        # Check that field tensors were created
        assert len(basic_leaf_cube.field_tensors) == 2
        assert "U" in basic_leaf_cube.field_tensors
        assert "p" in basic_leaf_cube.field_tensors

        # Check tensor shapes and properties
        u_tensor = basic_leaf_cube.field_tensors["U"]
        assert u_tensor.width == 8
        assert u_tensor.bnd == 2
        assert u_tensor.raw.shape == (
            12,
            12,
            12,
            3,
        )  # (width + 2*bnd, width + 2*bnd, width + 2*bnd, channels)
        assert u_tensor.raw.dtype == torch.float32

    def test_field_access_inheritance(
        self,
        basic_leaf_cube: LeafCube,
        field_data_dict: dict[str, FieldDataAttribute],
    ):
        """Test that LeafCube inherits field access methods from Cube"""
        basic_leaf_cube.allocate_field_tensors(field_data_dict)

        # Test __getitem__ access
        _ = basic_leaf_cube["U"]

        # Test __getattr__ access
        _ = basic_leaf_cube.p

        # Test items method
        for name, field_tensor in basic_leaf_cube.items():
            assert name in field_data_dict
            assert isinstance(field_tensor, FieldTensor)

    def test_empty_face_ids(self):
        """Test LeafCube with empty face_ids"""
        cube_code = gen_cube_code(root_code=1, morton_code=1)
        face_ids = torch.tensor([], dtype=torch.int32)
        neighbor_codes = []

        leaf_cube = LeafCube(
            width=6,
            bnd_width=1,
            depth=1,
            cube_code=cube_code,
            face_ids=face_ids,
            neighbor_codes=neighbor_codes,
        )

        assert torch.equal(leaf_cube.face_ids, face_ids)
        assert leaf_cube.neighbor_codes == neighbor_codes
        assert len(leaf_cube.face_ids) == 0
        assert len(leaf_cube.neighbor_codes) == 0

    def test_empty_neighbor_keys(self):
        """Test LeafCube with empty neighbor_keys"""
        cube_code = gen_cube_code(root_code=2, morton_code=2)
        face_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
        neighbor_codes = []

        leaf_cube = LeafCube(
            width=8,
            bnd_width=2,
            depth=2,
            cube_code=cube_code,
            face_ids=face_ids,
            neighbor_codes=neighbor_codes,
        )

        assert torch.equal(leaf_cube.face_ids, face_ids)
        assert leaf_cube.neighbor_codes == neighbor_codes
        assert len(leaf_cube.face_ids) == 3
        assert len(leaf_cube.neighbor_codes) == 0
