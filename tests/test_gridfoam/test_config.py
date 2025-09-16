import pathlib
from typing import Any

import pytest
import torch
import yaml
from pydantic import ValidationError

from gridfoam.config import (
    Config,
    CubeConfig,
    IoConfig,
    MeshConfig,
    device_validator,
)
from gridfoam.utils.enums import GridMode


class TestDeviceValidator:
    """Test device_validator function"""

    def test_device_validator_cpu_string(self):
        """Test device_validator with 'cpu' string"""
        result = device_validator("cpu")
        assert result == torch.device("cpu")


    def test_device_validator_cuda_multiple_devices(self):
        """Test device_validator with different CUDA devices"""
        for i in range(4):
            result = device_validator(f"cuda:{i}")
            assert result == torch.device(f"cuda:{i}")

    def test_device_validator_torch_device_cpu(self):
        """Test device_validator with torch.device('cpu')"""
        device = torch.device("cpu")
        result = device_validator(device)
        assert result == device

    def test_device_validator_torch_device_cuda(self):
        """Test device_validator with torch.device('cuda:0')"""
        device = torch.device("cuda:0")
        result = device_validator(device)
        assert result == device

    def test_device_validator_invalid_string(self):
        """Test device_validator with invalid string"""
        with pytest.raises(
            ValueError, match="device must be 'cpu' or 'cuda:<int>'"
        ):
            device_validator("invalid")

    def test_device_validator_invalid_cuda_format(self):
        """Test device_validator with invalid CUDA format"""
        invalid_formats = ["cuda", "cuda:", "cuda:abc", "cuda:-1"]
        for invalid_format in invalid_formats:
            with pytest.raises(
                ValueError, match="device must be 'cpu' or 'cuda:<int>'"
            ):
                device_validator(invalid_format)


class TestMeshConfig:
    """Test MeshConfig class"""

    def test_mesh_config_valid(self):
        """Test MeshConfig with valid parameters"""
        mesh_file = pathlib.Path("tests/data/stl/bunny.stl")
        config = MeshConfig(file=mesh_file)
        assert config.file == mesh_file

    def test_mesh_config_frozen(self):
        """Test that MeshConfig is frozen (immutable)"""
        mesh_file = pathlib.Path("tests/data/stl/bunny.stl")
        config = MeshConfig(file=mesh_file)

        with pytest.raises(ValidationError):
            config.file = pathlib.Path("different/path.stl")


class TestCubeConfig:
    """Test CubeConfig class"""

    def test_cube_config_valid(self):
        """Test CubeConfig with valid parameters"""
        config = CubeConfig(width=8, bnd_width=2)
        assert config.width == 8
        assert config.bnd_width == 2

    def test_cube_config_different_values(self):
        """Test CubeConfig with different values"""
        config = CubeConfig(width=16, bnd_width=4)
        assert config.width == 16
        assert config.bnd_width == 4

    def test_cube_config_invalid_width(self):
        """Test CubeConfig with zero values"""
        with pytest.raises(ValidationError):
            CubeConfig(width=0, bnd_width=2)

    def test_cube_config_invalid_bnd_width(self):
        """Test CubeConfig with negative values"""
        with pytest.raises(ValidationError):
            CubeConfig(width=8, bnd_width=0)

    def test_cube_config_frozen(self):
        """Test that CubeConfig is frozen (immutable)"""
        config = CubeConfig(width=8, bnd_width=2)

        with pytest.raises(ValidationError):
            config.width = 16


class TestIoConfig:
    """Test IoConfig class"""

    def test_io_config_valid(self):
        """Test IoConfig with valid parameters"""
        output_dir = pathlib.Path("tests/outputs")
        config = IoConfig(
            output_dir=output_dir,
            mode=GridMode.CELL,
            only_leaves=True,
            overwrite_file=False,
        )
        assert config.output_dir == output_dir
        assert config.mode == GridMode.CELL
        assert config.only_leaves is True
        assert config.overwrite_file is False

    def test_io_config_cube_mode(self):
        """Test IoConfig with CUBE mode"""
        output_dir = pathlib.Path("outputs")
        config = IoConfig(
            output_dir=output_dir,
            mode=GridMode.CUBE,
            only_leaves=False,
            overwrite_file=True,
        )
        assert config.output_dir == output_dir
        assert config.mode == GridMode.CUBE
        assert config.only_leaves is False
        assert config.overwrite_file is True


    def test_io_config_frozen(self):
        """Test that IoConfig is frozen (immutable)"""
        output_dir = pathlib.Path("tests/outputs")
        config = IoConfig(
            output_dir=output_dir,
            mode=GridMode.CELL,
            only_leaves=True,
            overwrite_file=False,
        )

        with pytest.raises(ValidationError):
            config.mode = GridMode.CUBE


class TestConfig:
    """Test Config class"""

    @pytest.fixture
    def valid_config_data(self) -> dict[str, Any]:
        """Valid configuration data for testing"""
        return {
            "io": {
                "output_dir": "tests/outputs",
                "mode": "cell",
                "only_leaves": True,
                "overwrite_file": False,
            },
            "mesh": {
                "file": "tests/data/stl/bunny.stl",
            },
            "cube": {
                "width": 8,
                "bnd_width": 2,
            },
            "device": "cpu",
        }

    def test_config_valid(self, valid_config_data: dict[str, Any]):
        """Test Config with valid parameters"""
        config = Config.model_validate(valid_config_data)

        assert config.io.output_dir == pathlib.Path("tests/outputs")
        assert config.io.mode == GridMode.CELL
        assert config.io.only_leaves is True
        assert config.io.overwrite_file is False

        assert config.mesh.file == pathlib.Path("tests/data/stl/bunny.stl")

        assert config.cube.width == 8
        assert config.cube.bnd_width == 2

        assert config.device == torch.device("cpu")

    def test_config_with_cuda_device(self, valid_config_data: dict[str, Any]):
        """Test Config with CUDA device"""
        valid_config_data["device"] = "cuda:0"
        config = Config.model_validate(valid_config_data)
        assert config.device == torch.device("cuda:0")

    def test_config_default_device(self, valid_config_data: dict[str, Any]):
        """Test Config with default device (no device specified)"""
        del valid_config_data["device"]
        config = Config.model_validate(valid_config_data)
        assert config.device == torch.device("cpu")


    def test_config_missing_required_fields(self):
        """Test Config with missing required fields"""
        incomplete_data = {
            "io": {
                "output_dir": "tests/outputs",
                "mode": "cell",
                "only_leaves": True,
                "overwrite_file": False,
            },
            # Missing mesh and cube
        }

        with pytest.raises(ValidationError):
            Config.model_validate(incomplete_data)

    def test_config_invalid_device(self, valid_config_data: dict[str, Any]):
        """Test Config with invalid device"""
        valid_config_data["device"] = "invalid_device"

        with pytest.raises(ValidationError):
            Config.model_validate(valid_config_data)

    def test_config_invalid_mode(self, valid_config_data: dict[str, Any]):
        """Test Config with invalid mode"""
        valid_config_data["io"]["mode"] = "invalid_mode"

        with pytest.raises(ValidationError):
            Config.model_validate(valid_config_data)

    def test_config_frozen(self, valid_config_data: dict[str, Any]):
        """Test that Config is frozen (immutable)"""
        config = Config.model_validate(valid_config_data)

        with pytest.raises(ValidationError):
            config.device = torch.device("cuda:0")

    def test_config_from_file(self):
        """Test Config.model_validate_file method"""
        config_path = pathlib.Path("tests/data/yaml/bunny.yaml")
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = Config.model_validate(config_dict)

        assert isinstance(config, Config)
        assert config.io.output_dir == pathlib.Path("tests/outputs/grid")
        assert config.mesh.file == pathlib.Path("tests/data/stl/bunny.stl")
        assert config.cube.width == 8
        assert config.cube.bnd_width == 2
        assert config.device == torch.device("cpu")
