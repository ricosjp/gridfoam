import pytest
import torch
from pydantic import ValidationError

from gridfoam.settings import (
    CubeSetting,
    FieldDataAttribute,
    GridSetting,
    device_validator,
    dtype_validator,
)
from gridfoam.utils.enums import Constants


class TestDeviceValidator:
    """Test device_validator function"""

    def test_device_validator_cpu(self):
        """Test device_validator with 'cpu'"""
        result = device_validator("cpu")
        assert result == torch.device("cpu")

    def test_device_validator_cuda_valid(self):
        """Test device_validator with valid CUDA device"""
        result = device_validator("cuda:0")
        assert result == torch.device("cuda:0")

        result = device_validator("cuda:1")
        assert result == torch.device("cuda:1")

        result = device_validator("cuda:999")
        assert result == torch.device("cuda:999")

    def test_device_validator_cuda_invalid(self):
        """Test device_validator with invalid CUDA device"""
        with pytest.raises(
            ValueError,
            match="device must be 'cpu' or 'cuda:<int>' \\(e.g. 'cuda:0'\\)",
        ):
            device_validator("cuda")

        with pytest.raises(
            ValueError,
            match="device must be 'cpu' or 'cuda:<int>' \\(e.g. 'cuda:0'\\)",
        ):
            device_validator("cuda:")

        with pytest.raises(
            ValueError,
            match="device must be 'cpu' or 'cuda:<int>' \\(e.g. 'cuda:0'\\)",
        ):
            device_validator("cuda:abc")

        with pytest.raises(
            ValueError,
            match="device must be 'cpu' or 'cuda:<int>' \\(e.g. 'cuda:0'\\)",
        ):
            device_validator("gpu:0")


class TestDtypeValidator:
    """Test dtype_validator function"""

    def test_dtype_validator_valid_types(self):
        """Test dtype_validator with valid dtype strings"""
        test_cases = [
            ("bool", torch.bool),
            ("int32", torch.int32),
            ("int64", torch.int64),
            ("float32", torch.float32),
            ("float64", torch.float64),
        ]

        for dtype_str, expected_dtype in test_cases:
            result = dtype_validator(dtype_str)
            assert result == expected_dtype

    def test_dtype_validator_invalid_type(self):
        """Test dtype_validator with invalid dtype string"""
        with pytest.raises(
            ValueError, match="Invalid dtype string: invalid_dtype"
        ):
            dtype_validator("invalid_dtype")


class TestFieldDataAttribute:
    """Test FieldDataAttribute class"""

    def test_field_data_attribute_valid(self):
        """Test FieldDataAttribute with valid parameters"""
        attr = FieldDataAttribute(shape=(3,), dtype=torch.float32)
        assert attr.shape == (3,)
        assert attr.dtype == torch.float32

    def test_field_data_attribute_complex_shape(self):
        """Test FieldDataAttribute with complex shape"""
        attr = FieldDataAttribute(shape=(2, 3, 4), dtype=torch.int64)
        assert attr.shape == (2, 3, 4)
        assert attr.dtype == torch.int64

    def test_field_data_attribute_invalid_shape_zero(self):
        """Test FieldDataAttribute with zero in shape"""
        with pytest.raises(ValidationError):
            FieldDataAttribute(shape=(0,), dtype=torch.float32)

    def test_field_data_attribute_invalid_shape_negative(self):
        """Test FieldDataAttribute with negative value in shape"""
        with pytest.raises(ValidationError):
            FieldDataAttribute(shape=(3, -1), dtype=torch.float32)


class TestCubeSetting:
    """Test CubeSetting class"""

    def test_cube_setting_custom_values(self):
        """Test CubeSetting with custom values"""
        cube_setting = CubeSetting(width=12, bnd_width=1)
        assert cube_setting.width == 12
        assert cube_setting.bnd_width == 1

    def test_cube_setting_width_too_small(self):
        """Test CubeSetting with width too small"""
        with pytest.raises(ValidationError):
            CubeSetting(width=7)

    def test_cube_setting_width_too_large(self):
        """Test CubeSetting with width too large"""
        with pytest.raises(ValidationError):
            CubeSetting(width=17)

    def test_cube_setting_invalid_bnd_width(self):
        """Test CubeSetting with invalid bnd_width"""
        with pytest.raises(ValidationError):
            CubeSetting(bnd_width=3)


class TestGridSetting:
    """Test GridSetting class"""

    @pytest.fixture
    def basic_grid_setting(self):
        """Create a basic GridSetting for testing"""
        return GridSetting(
            blockXMin=0.0,
            blockXMax=1.0,
            blockYMin=0.0,
            blockYMax=1.0,
            nBlockX=10,
            nBlockY=10,
        )

    def test_grid_setting_basic(self, basic_grid_setting: GridSetting):
        """Test GridSetting with basic parameters"""
        assert basic_grid_setting.blockXMin == 0.0
        assert basic_grid_setting.blockXMax == 1.0
        assert basic_grid_setting.blockYMin == 0.0
        assert basic_grid_setting.blockYMax == 1.0
        assert basic_grid_setting.blockZMin == 0.0  # default
        assert basic_grid_setting.blockZMax == 1.0  # default
        assert basic_grid_setting.nBlockX == 10
        assert basic_grid_setting.nBlockY == 10
        assert basic_grid_setting.nBlockZ == 1  # default
        assert basic_grid_setting.alpha == 0.3  # default
        assert basic_grid_setting.depth_limit == 5  # default
        assert basic_grid_setting.device == torch.device("cpu")  # default

    def test_grid_setting_custom_values(self):
        """Test GridSetting with custom values"""
        grid_setting = GridSetting(
            blockXMin=-1.0,
            blockXMax=2.0,
            blockYMin=-0.5,
            blockYMax=1.5,
            blockZMin=-2.0,
            blockZMax=3.0,
            nBlockX=20,
            nBlockY=15,
            nBlockZ=5,
            alpha=0.4,
            depth_limit=10,
            device=torch.device("cuda:0"),
        )
        assert grid_setting.blockXMin == -1.0
        assert grid_setting.blockXMax == 2.0
        assert grid_setting.blockYMin == -0.5
        assert grid_setting.blockYMax == 1.5
        assert grid_setting.blockZMin == -2.0
        assert grid_setting.blockZMax == 3.0
        assert grid_setting.nBlockX == 20
        assert grid_setting.nBlockY == 15
        assert grid_setting.nBlockZ == 5
        assert grid_setting.alpha == 0.4
        assert grid_setting.depth_limit == 10
        assert grid_setting.device == torch.device("cuda:0")

    def test_grid_setting_alpha_validation(self):
        """Test GridSetting alpha validation"""
        # Valid alpha
        grid_setting = GridSetting(
            blockXMin=0.0,
            blockXMax=1.0,
            blockYMin=0.0,
            blockYMax=1.0,
            nBlockX=10,
            nBlockY=10,
            alpha=0.5,
        )
        assert grid_setting.alpha == 0.5

        # Invalid alpha (too large)
        with pytest.raises(ValidationError):
            GridSetting(
                blockXMin=0.0,
                blockXMax=1.0,
                blockYMin=0.0,
                blockYMax=1.0,
                nBlockX=10,
                nBlockY=10,
                alpha=0.6,
            )

    def test_grid_setting_depth_limit_validation(self):
        """Test GridSetting depth_limit validation"""
        # Valid depth_limit
        grid_setting = GridSetting(
            blockXMin=0.0,
            blockXMax=1.0,
            blockYMin=0.0,
            blockYMax=1.0,
            nBlockX=10,
            nBlockY=10,
            depth_limit=Constants.MAX_OCTREE_DEPTH,
        )
        assert grid_setting.depth_limit == Constants.MAX_OCTREE_DEPTH

        # Invalid depth_limit (too large)
        with pytest.raises(ValidationError):
            GridSetting(
                blockXMin=0.0,
                blockXMax=1.0,
                blockYMin=0.0,
                blockYMax=1.0,
                nBlockX=10,
                nBlockY=10,
                depth_limit=Constants.MAX_OCTREE_DEPTH + 1,
            )

        # Invalid depth_limit (too small)
        with pytest.raises(ValidationError):
            GridSetting(
                blockXMin=0.0,
                blockXMax=1.0,
                blockYMin=0.0,
                blockYMax=1.0,
                nBlockX=10,
                nBlockY=10,
                depth_limit=0,
            )

    def test_grid_setting_cube_setting(self):
        """Test GridSetting cube_setting"""
        custom_cube_setting = CubeSetting(width=12, bnd_width=1)
        grid_setting = GridSetting(
            blockXMin=0.0,
            blockXMax=1.0,
            blockYMin=0.0,
            blockYMax=1.0,
            nBlockX=10,
            nBlockY=10,
            cube_setting=custom_cube_setting,
        )
        assert grid_setting.cube_setting == custom_cube_setting

    def test_grid_setting_field_data_dict_defaults(
        self, basic_grid_setting: GridSetting
    ):
        """Test GridSetting field_data_dict default values"""
        field_data = basic_grid_setting.field_data_dict
        assert "U" in field_data
        assert "p" in field_data
        assert field_data["U"].shape == (3,)
        assert field_data["U"].dtype == torch.float32
        assert field_data["p"].shape == (1,)
        assert field_data["p"].dtype == torch.float32

    def test_grid_setting_field_data_dict_custom(self):
        """Test GridSetting with custom field_data_dict"""
        custom_field_data = {
            "F": FieldDataAttribute(shape=(3,), dtype=torch.float64),
            "T": FieldDataAttribute(shape=(1,), dtype=torch.float32),
        }
        grid_setting = GridSetting(
            blockXMin=0.0,
            blockXMax=1.0,
            blockYMin=0.0,
            blockYMax=1.0,
            nBlockX=10,
            nBlockY=10,
            field_data_dict=custom_field_data,
        )
        field_data = grid_setting.field_data_dict
        assert "F" in field_data
        assert "T" in field_data
        assert field_data["F"].shape == (3,)
        assert field_data["F"].dtype == torch.float64
        assert field_data["T"].shape == (1,)
        assert field_data["T"].dtype == torch.float32

    def test_grid_setting_get_domain(self, basic_grid_setting: GridSetting):
        """Test GridSetting get_domain method"""
        domain = basic_grid_setting.get_domain()
        assert domain.min.shape == (3,)
        assert domain.max.shape == (3,)
        assert torch.equal(domain.min, torch.tensor([0.0, 0.0, 0.0]))
        assert torch.equal(domain.max, torch.tensor([1.0, 1.0, 1.0]))

    def test_grid_setting_get_domain_custom_z(self):
        """Test GridSetting get_domain method with custom Z values"""
        grid_setting = GridSetting(
            blockXMin=0.0,
            blockXMax=1.0,
            blockYMin=0.0,
            blockYMax=1.0,
            blockZMin=-2.0,
            blockZMax=3.0,
            nBlockX=10,
            nBlockY=10,
        )
        domain = grid_setting.get_domain()
        assert torch.equal(domain.min, torch.tensor([0.0, 0.0, -2.0]))
        assert torch.equal(domain.max, torch.tensor([1.0, 1.0, 3.0]))

    def test_grid_setting_get_block_divisions(
        self, basic_grid_setting: GridSetting
    ):
        """Test GridSetting get_block_divisions method"""
        divisions = basic_grid_setting.get_block_divisions()
        assert divisions.shape == (3,)
        assert divisions.dtype == torch.int32
        assert torch.equal(
            divisions, torch.tensor([10, 10, 1], dtype=torch.int32)
        )

    def test_grid_setting_get_block_divisions_custom(self):
        """Test GridSetting get_block_divisions method with custom values"""
        grid_setting = GridSetting(
            blockXMin=0.0,
            blockXMax=1.0,
            blockYMin=0.0,
            blockYMax=1.0,
            nBlockX=20,
            nBlockY=15,
            nBlockZ=5,
        )
        divisions = grid_setting.get_block_divisions()
        assert torch.equal(
            divisions, torch.tensor([20, 15, 5], dtype=torch.int32)
        )
