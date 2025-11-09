import pathlib
import re
from typing import Annotated

import torch
from pydantic import BaseModel, Field, PlainValidator, field_validator

from gridfoam.utils.enums import DiscretizationMode, GridOutputMode, SolverName


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
    mode: GridOutputMode = Field(default=GridOutputMode.CELL)
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


class SchemeChoice(BaseModel, frozen=True):
    mode: DiscretizationMode | None = DiscretizationMode.IMPLICIT
    """
    mode : DiscretizationMode | None
        Mode of the discretization.
        If None, the mode is implicit.
    """
    scheme: str
    """
    scheme : str
        Scheme for the discretization.
    """


class fvSchemesConfig(BaseModel, frozen=True):
    ddtSchemes: dict[str, SchemeChoice]
    """
    ddtSchemes : dict[str, SchemeChoice]
        Scheme for the ddt.
    """
    divSchemes: dict[str, SchemeChoice] | None = None
    """
    divSchemes : dict[str, SchemeChoice] | None
        Scheme for the div.
    """
    laplacianSchemes: dict[str, SchemeChoice] | None = None
    """
    laplacianSchemes : dict[str, SchemeChoice] | None
        Scheme for the laplacian.
    """

    @field_validator(
        "ddtSchemes", "divSchemes", "laplacianSchemes", mode="before"
    )
    @classmethod
    def regularize_keys(
        cls, d: dict[str, SchemeChoice] | None
    ) -> dict[str, SchemeChoice] | None:
        if d is None:
            return None
        return {re.sub(r"\s*,\s*", ", ", k): v for k, v in d.items()}


class SolverChoice(BaseModel, frozen=True):
    type: SolverName
    """
    type : SolverName
        Type of the solver.
    """
    preconditioner: str | None = None
    """
    preconditioner : str | None
        Preconditioner for the solver.
        If None, no preconditioner is used.
    """
    tolerance: float = Field(default=1e-6)
    """
    tolerance : float
        Tolerance for the solver.
    """
    rel_tolerance: float = Field(default=1e-6)
    """
    rel_tolerance : float
        Relative tolerance for the solver.
    """
    max_iter: int = Field(default=1000)
    """
    max_iter : int
        Maximum number of iterations for the solver.
    """


class fvSolutionConfig(BaseModel, frozen=True):
    solvers: dict[str, SolverChoice]
    """
    solvers : dict[str, SolverChoice]
        Solvers configuration.
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
    fvSolution: fvSolutionConfig
    """
    fvSolution : fvSolutionConfig
        Fv solution configuration.
    """


class GridfoamConfig(BaseModel, frozen=True):
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


class YamlRoot(BaseModel, frozen=True):
    gridfoam: GridfoamConfig
    """
    gridfoam : Config
        Gridfoam configuration.
    """
