import pathlib

import pytest
import torch
from pydantic import ValidationError

from gridfoam.config import (
    Config,
    ControlConfig,
    CubeConfig,
    DdtConfig,
    DivConfig,
    IoConfig,
    MeshConfig,
    SimulatorConfig,
    device_validator,
    fvSchemesConfig,
)
from gridfoam.utils.enums import GridMode


class TestDeviceValidator:
    """Test device_validator function."""

    def test_valid_cpu_string(self):
        """Test that 'cpu' string is converted to torch.device('cpu')."""
        result = device_validator("cpu")
        assert result == torch.device("cpu")

    def test_valid_cuda_string(self):
        """Test that 'cuda:0' string is converted to torch.device('cuda:0')."""
        result = device_validator("cuda:0")
        assert result == torch.device("cuda:0")

    def test_valid_cuda_string_with_different_id(self):
        """Test that 'cuda:1' string is converted to torch.device('cuda:1')."""
        result = device_validator("cuda:1")
        assert result == torch.device("cuda:1")

    def test_valid_torch_device(self):
        """Test that torch.device object is returned as is."""
        device = torch.device("cpu")
        result = device_validator(device)
        assert result == device

    def test_invalid_device_string(self):
        """Test that invalid device string raises ValueError."""
        with pytest.raises(
            ValueError, match="device must be 'cpu' or 'cuda:<int>'"
        ):
            device_validator("invalid")

    def test_invalid_cuda_format(self):
        """Test that invalid cuda format raises ValueError."""
        with pytest.raises(
            ValueError, match="device must be 'cpu' or 'cuda:<int>'"
        ):
            device_validator("cuda")

    def test_invalid_cuda_with_float(self):
        """Test that cuda with float raises ValueError."""
        with pytest.raises(
            ValueError, match="device must be 'cpu' or 'cuda:<int>'"
        ):
            device_validator("cuda:1.0")


class TestMeshConfig:
    """Test MeshConfig class."""

    def test_valid_mesh_config(self):
        """Test creating valid MeshConfig."""
        config = MeshConfig(file=pathlib.Path("test.obj"))
        assert config.file == pathlib.Path("test.obj")

    def test_mesh_config_frozen(self):
        """Test that MeshConfig is frozen."""
        config = MeshConfig(file=pathlib.Path("test.obj"))
        with pytest.raises(ValidationError):
            config.file = pathlib.Path("new.obj")


class TestCubeConfig:
    """Test CubeConfig class."""

    def test_default_values(self):
        """Test default values for CubeConfig."""
        config = CubeConfig()
        assert config.interior_width == 8
        assert config.halo_width == 2

    def test_custom_values(self):
        """Test custom values for CubeConfig."""
        config = CubeConfig(interior_width=16, halo_width=4)
        assert config.interior_width == 16
        assert config.halo_width == 4

    def test_minimum_values(self):
        """Test minimum values for CubeConfig."""
        config = CubeConfig(interior_width=8, halo_width=2)
        assert config.interior_width == 8
        assert config.halo_width == 2

    def test_interior_width_too_small(self):
        """Test that interior_width < 8 raises ValidationError."""
        with pytest.raises(ValidationError):
            CubeConfig(interior_width=7)

    def test_halo_width_too_small(self):
        """Test that halo_width < 2 raises ValidationError."""
        with pytest.raises(ValidationError):
            CubeConfig(halo_width=1)

    def test_cube_config_frozen(self):
        """Test that CubeConfig is frozen."""
        config = CubeConfig()
        with pytest.raises(ValidationError):
            config.interior_width = 16


class TestIoConfig:
    """Test IoConfig class."""

    def test_valid_io_config(self):
        """Test creating valid IoConfig."""
        config = IoConfig(
            output_dir=pathlib.Path("/tmp/output"),
            mode=GridMode.CELL,
            only_leaves=True,
            overwrite_file=True,
        )
        assert config.output_dir == pathlib.Path("/tmp/output")
        assert config.mode == GridMode.CELL
        assert config.only_leaves is True
        assert config.overwrite_file is True

    def test_default_values(self):
        """Test default values for IoConfig."""
        config = IoConfig(output_dir=pathlib.Path("/tmp/output"))
        assert config.mode == GridMode.CELL
        assert config.only_leaves is True
        assert config.overwrite_file is True

    def test_cube_mode(self):
        """Test IoConfig with CUBE mode."""
        config = IoConfig(
            output_dir=pathlib.Path("/tmp/output"), mode=GridMode.CUBE
        )
        assert config.mode == GridMode.CUBE

    def test_io_config_frozen(self):
        """Test that IoConfig is frozen."""
        config = IoConfig(output_dir=pathlib.Path("/tmp/output"))
        with pytest.raises(ValidationError):
            config.mode = GridMode.CUBE


class TestDdtConfig:
    """Test DdtConfig class."""

    def test_default_value(self):
        """Test default value for DdtConfig."""
        config = DdtConfig()
        assert config.offset_coefficient == 0.9

    def test_custom_value(self):
        """Test custom value for DdtConfig."""
        config = DdtConfig(offset_coefficient=0.5)
        assert config.offset_coefficient == 0.5

    def test_boundary_values(self):
        """Test boundary values for DdtConfig."""
        config_min = DdtConfig(offset_coefficient=0.0)
        config_max = DdtConfig(offset_coefficient=1.0)
        assert config_min.offset_coefficient == 0.0
        assert config_max.offset_coefficient == 1.0

    def test_offset_coefficient_too_small(self):
        """Test that offset_coefficient < 0.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            DdtConfig(offset_coefficient=-0.1)

    def test_offset_coefficient_too_large(self):
        """Test that offset_coefficient > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            DdtConfig(offset_coefficient=1.1)

    def test_ddt_config_frozen(self):
        """Test that DdtConfig is frozen."""
        config = DdtConfig()
        with pytest.raises(ValidationError):
            config.offset_coefficient = 0.5


class TestDivConfig:
    """Test DivConfig class."""

    def test_valid_div_config(self):
        """Test creating valid DivConfig."""
        config = DivConfig(scheme="Gauss linear")
        assert config.scheme == "Gauss linear"

    def test_div_config_frozen(self):
        """Test that DivConfig is frozen."""
        config = DivConfig(scheme="Gauss linear")
        with pytest.raises(ValidationError):
            config.scheme = "Gauss upwind"


class TestControlConfig:
    """Test ControlConfig class."""

    def test_valid_control_config(self):
        """Test creating valid ControlConfig."""
        config = ControlConfig(
            deltaT=0.001,
            endTime=1.0,
            writeInterval=100,
        )
        assert config.deltaT == 0.001
        assert config.endTime == 1.0
        assert config.writeInterval == 100

    def test_control_config_frozen(self):
        """Test that ControlConfig is frozen."""
        config = ControlConfig(deltaT=0.001, endTime=1.0, writeInterval=100)
        with pytest.raises(ValidationError):
            config.deltaT = 0.002


class TestFvSchemesConfig:
    """Test fvSchemesConfig class."""

    def test_valid_fv_schemes_config(self):
        """Test creating valid fvSchemesConfig."""
        ddt_config = DdtConfig()
        div_config = DivConfig(scheme="Gauss linear")
        config = fvSchemesConfig(ddt=ddt_config, div=div_config)
        assert config.ddt == ddt_config
        assert config.div == div_config

    def test_fv_schemes_config_frozen(self):
        """Test that fvSchemesConfig is frozen."""
        ddt_config = DdtConfig()
        div_config = DivConfig(scheme="Gauss linear")
        config = fvSchemesConfig(ddt=ddt_config, div=div_config)
        with pytest.raises(ValidationError):
            config.ddt = DdtConfig(offset_coefficient=0.5)


class TestSimulatorConfig:
    """Test SimulatorConfig class."""

    def test_valid_simulator_config(self):
        """Test creating valid SimulatorConfig."""
        control_config = ControlConfig(
            deltaT=0.001,
            endTime=1.0,
            writeInterval=100,
        )
        ddt_config = DdtConfig()
        div_config = DivConfig(scheme="Gauss linear")
        fv_schemes_config = fvSchemesConfig(ddt=ddt_config, div=div_config)
        config = SimulatorConfig(
            control=control_config, fvSchemes=fv_schemes_config
        )
        assert config.control == control_config
        assert config.fvSchemes == fv_schemes_config

    def test_simulator_config_frozen(self):
        """Test that SimulatorConfig is frozen."""
        control_config = ControlConfig(
            deltaT=0.001,
            endTime=1.0,
            writeInterval=100,
        )
        ddt_config = DdtConfig()
        div_config = DivConfig(scheme="Gauss linear")
        fv_schemes_config = fvSchemesConfig(ddt=ddt_config, div=div_config)
        config = SimulatorConfig(
            control=control_config, fvSchemes=fv_schemes_config
        )
        with pytest.raises(ValidationError):
            config.control = ControlConfig(
                deltaT=0.002, endTime=1.0, writeInterval=100
            )


class TestConfig:
    """Test Config class."""

    def test_valid_config(self):
        """Test creating valid Config."""
        control_config = ControlConfig(
            deltaT=0.001,
            endTime=1.0,
            writeInterval=100,
        )
        ddt_config = DdtConfig()
        div_config = DivConfig(scheme="Gauss linear")
        fv_schemes_config = fvSchemesConfig(ddt=ddt_config, div=div_config)
        simulator_config = SimulatorConfig(
            control=control_config, fvSchemes=fv_schemes_config
        )
        io_config = IoConfig(output_dir=pathlib.Path("/tmp/output"))
        mesh_config = MeshConfig(file=pathlib.Path("test.obj"))
        cube_config = CubeConfig()

        config = Config(
            simulator=simulator_config,
            io=io_config,
            mesh=mesh_config,
            cube=cube_config,
        )
        assert config.simulator == simulator_config
        assert config.io == io_config
        assert config.mesh == mesh_config
        assert config.cube == cube_config
        assert config.device == torch.device("cpu")

    def test_config_frozen(self):
        """Test that Config is frozen."""
        control_config = ControlConfig(
            deltaT=0.001,
            endTime=1.0,
            writeInterval=100,
        )
        ddt_config = DdtConfig()
        div_config = DivConfig(scheme="Gauss linear")
        fv_schemes_config = fvSchemesConfig(ddt=ddt_config, div=div_config)
        simulator_config = SimulatorConfig(
            control=control_config, fvSchemes=fv_schemes_config
        )
        io_config = IoConfig(output_dir=pathlib.Path("/tmp/output"))
        mesh_config = MeshConfig(file=pathlib.Path("test.obj"))
        cube_config = CubeConfig()

        config = Config(
            simulator=simulator_config,
            io=io_config,
            mesh=mesh_config,
            cube=cube_config,
        )
        with pytest.raises(ValidationError):
            config.device = torch.device("cuda:0")
