"""Tests for config module."""

import pathlib

import pytest
import torch

from gridfoam.DNA.config import (
    ControlConfig,
    CubeConfig,
    GridfoamConfig,
    IoConfig,
    MeshConfig,
    SimulatorConfig,
    YamlRoot,
    device_validator,
    fvSchemesConfig,
    fvSolutionConfig,
    SolverChoice,
)
from gridfoam.DNA.enum import NormType
from gridfoam.DNA.scheme.fvm.ddt._choice import FVMDdtSchemeChoice
from gridfoam.DNA.scheme.fvm.div._choice import FVMDivSchemeChoice
from gridfoam.DNA.scheme.fvm.grad._choice import FVMGradSchemeChoice
from gridfoam.DNA.scheme.fvm.laplacian._choice import FVMLaplacianSchemeChoice
from gridfoam.DNA.scheme.solver._choice import SolverMethodChoice


def test_device_validator_cpu():
    """Test device_validator with cpu string."""
    result = device_validator("cpu")
    assert result == torch.device("cpu")


def test_device_validator_cuda():
    """Test device_validator with cuda string."""
    result = device_validator("cuda:0")
    assert result == torch.device("cuda:0")


def test_device_validator_torch_device():
    """Test device_validator with torch.device."""
    device = torch.device("cpu")
    result = device_validator(device)
    assert result == device


def test_device_validator_invalid():
    """Test device_validator raises error for invalid input."""
    with pytest.raises(ValueError, match="device must be"):
        device_validator("invalid")


def test_mesh_config():
    """Test MeshConfig creation."""
    config = MeshConfig(file=pathlib.Path("test.stl"))
    assert config.file == pathlib.Path("test.stl")


def test_cube_config_defaults():
    """Test CubeConfig with defaults."""
    config = CubeConfig()
    assert config.interior_width == 8
    assert config.halo_width == 2
    assert config.device == torch.device("cpu")


def test_cube_config_custom():
    """Test CubeConfig with custom values."""
    config = CubeConfig(
        interior_width=16, halo_width=4, device=torch.device("cpu")
    )
    assert config.interior_width == 16
    assert config.halo_width == 4


def test_cube_config_validation():
    """Test CubeConfig validation."""
    with pytest.raises(Exception):  # Pydantic validation error
        CubeConfig(interior_width=4)  # Less than minimum 8


def test_io_config():
    """Test IoConfig creation."""
    config = IoConfig(
        output_dir=pathlib.Path("output"),
        base_name="test",
        only_leaves=True,
        overwrite_file=True,
    )
    assert config.output_dir == pathlib.Path("output")
    assert config.base_name == "test"
    assert config.only_leaves is True
    assert config.overwrite_file is True


def test_control_config():
    """Test ControlConfig creation."""
    config = ControlConfig(deltaT=0.1, endTime=10.0, writeInterval=10)
    assert config.deltaT == 0.1
    assert config.endTime == 10.0
    assert config.writeInterval == 10


def test_fv_schemes_config():
    """Test fvSchemesConfig creation."""
    config = fvSchemesConfig(
        ddtSchemes={"default": FVMDdtSchemeChoice.EULER},
        divSchemes={"default": FVMDivSchemeChoice.UPWIND},
        laplacianSchemes={"default": FVMLaplacianSchemeChoice.LINEAR},
        gradSchemes={"default": FVMGradSchemeChoice.LINEAR},
    )
    assert config.ddtSchemes["default"] == FVMDdtSchemeChoice.EULER
    assert config.divSchemes["default"] == FVMDivSchemeChoice.UPWIND


def test_fv_schemes_config_regularize_keys():
    """Test fvSchemesConfig key regularization."""
    config = fvSchemesConfig(
        ddtSchemes={"default, test": FVMDdtSchemeChoice.EULER},
    )
    assert "default, test" in config.ddtSchemes


def test_solver_choice():
    """Test SolverChoice creation."""
    choice = SolverChoice(
        method=SolverMethodChoice.CG,
        tolerance=1e-6,
        rel_tolerance=1e-6,
        max_iter=1000,
        norm_type=NormType.L_2,
    )
    assert choice.method == SolverMethodChoice.CG
    assert choice.tolerance == 1e-6
    assert choice.norm_type == NormType.L_2


def test_fv_solution_config():
    """Test fvSolutionConfig creation."""
    solver_choice = SolverChoice(
        method=SolverMethodChoice.CG,
        tolerance=1e-6,
        rel_tolerance=1e-6,
        max_iter=1000,
        norm_type=NormType.L_2,
    )
    config = fvSolutionConfig(solvers={"poisson": solver_choice})
    assert "poisson" in config.solvers
    assert config.solvers["poisson"] == solver_choice


def test_simulator_config():
    """Test SimulatorConfig creation."""
    control = ControlConfig(deltaT=0.1, endTime=10.0, writeInterval=10)
    schemes = fvSchemesConfig()
    solution = fvSolutionConfig(solvers={})
    config = SimulatorConfig(
        control=control, fvSchemes=schemes, fvSolution=solution
    )
    assert config.control == control
    assert config.fvSchemes == schemes
    assert config.fvSolution == solution


def test_gridfoam_config():
    """Test GridfoamConfig creation."""
    simulator = SimulatorConfig(
        control=ControlConfig(deltaT=0.1, endTime=10.0, writeInterval=10),
        fvSchemes=fvSchemesConfig(),
        fvSolution=fvSolutionConfig(solvers={}),
    )
    io = IoConfig(output_dir=pathlib.Path("output"), base_name="test")
    mesh = MeshConfig(file=pathlib.Path("mesh.stl"))
    cube = CubeConfig()
    config = GridfoamConfig(simulator=simulator, io=io, mesh=mesh, cube=cube)
    assert config.simulator == simulator
    assert config.io == io
    assert config.mesh == mesh
    assert config.cube == cube


def test_yaml_root():
    """Test YamlRoot creation."""
    gridfoam_config = GridfoamConfig(
        simulator=SimulatorConfig(
            control=ControlConfig(deltaT=0.1, endTime=10.0, writeInterval=10),
            fvSchemes=fvSchemesConfig(),
            fvSolution=fvSolutionConfig(solvers={}),
        ),
        io=IoConfig(output_dir=pathlib.Path("output"), base_name="test"),
        mesh=MeshConfig(file=pathlib.Path("mesh.stl")),
        cube=CubeConfig(),
    )
    root = YamlRoot(gridfoam=gridfoam_config)
    assert root.gridfoam == gridfoam_config
