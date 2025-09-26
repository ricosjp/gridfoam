import pathlib
import re
from typing import Annotated

import torch
from pydantic import BaseModel, Field, PlainValidator

from gridfoam.utils.enums import GridMode


def device_validator(v: str | torch.device) -> torch.device:
    if isinstance(v, torch.device):
        return v
    if v == "cpu":
        return torch.device("cpu")
    if re.fullmatch(r"cuda:\d+", v):
        return torch.device(v)
    raise ValueError("device must be 'cpu' or 'cuda:<int>' (e.g. 'cuda:0')")


TorchDevice = Annotated[torch.device, PlainValidator(device_validator)]


class MeshConfig(BaseModel, frozen=True):
    file: pathlib.Path
    """
    file : pathlib.Path
        Path to the mesh file.
    """


class CubeConfig(BaseModel, frozen=True):
    interior_width: int = Field(default=8, ge=8)
    """
    interior_width : int
        Interior width of the cube.
    """
    halo_width: int = Field(default=2, ge=2)
    """
    halo_width : int
        Halo width of the cube
    """


class IoConfig(BaseModel, frozen=True):
    output_dir: pathlib.Path
    """
    output_dir : pathlib.Path
        Output directory for the grid.
    """
    mode: GridMode = Field(default=GridMode.CELL)
    """
    mode : GridMode
        Mode for saving the grid.
        - CELL: Save all grid data including cell data.
        - CUBE: Save only cube data, useful for checking the cube structure.
    """
    only_leaves: bool = Field(default=True)
    """
    only_leaves : bool
        Whether to save only leaf nodes.
        In the original grid,
        ghost nodes are introduced to share data between adjacent cells.
        However, for visualization purposes,
        saving only the leaf nodes is preferable,
        as including all nodes (including ghost nodes) can cause
        overlapping and make the visualization harder to interpret.
    """
    overwrite_file: bool = Field(default=True)
    """
    overwrite_file : bool
        Whether to overwrite the file if it already exists.
        Useful to avoid file corruption.
    """

class DdtConfig(BaseModel, frozen=True):
    offset_coefficient: float = Field(default=0.9, ge=0.0, le=1.0)
    """
    offset_coefficient : float, default=0.9
        Offset coefficient for the ddt.
        Set 0 for Euler implicit scheme.
        Set 1 for Crank-Nicolson scheme.
        Set 0.9 for default.
    """


class DivConfig(BaseModel, frozen=True):
    scheme: str
    """
    scheme : str
        Scheme for the div.
    """

class ControlConfig(BaseModel, frozen=True):
    deltaT: float
    """
    deltaT : float
        Time step for the simulation.
    """
    endTime: float
    """
    endTime : float
        Maximum time for the simulation.
    """
    writeInterval: int
    """
    writeInterval : int
        Write interval for the simulation.
    """

class fvSchemesConfig(BaseModel, frozen=True):
    ddt: DdtConfig
    """
    ddt : str
        Scheme for the fv.
    """
    div: DivConfig
    """
    div : DivConfig
        Scheme for the div.
    """


class SimulatorConfig(BaseModel, frozen=True):
    control: ControlConfig
    """
    control : ControlConfig
        Control configuration.
    """
    fvSchemes: fvSchemesConfig
    """
    fvSchemes : fvSchemesConfig
        Fv schemes configuration.
    """

class Config(BaseModel, frozen=True):
    simulator: SimulatorConfig
    """
    simulator : SimulatorConfig
        Simulator configuration.
    """

    io: IoConfig
    """
    io : IoConfig
        I/O configuration.
    """

    mesh: MeshConfig
    """
    mesh : MeshConfig
        Mesh configuration.
    """

    cube: CubeConfig
    """
    cube : CubeConfig
        Cube configuration.
    """

    device: TorchDevice = Field(default=torch.device("cpu"))
    """
    device : torch.device
        Device on which tensors are allocated.
    """
