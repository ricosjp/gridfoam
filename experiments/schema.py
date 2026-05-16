from __future__ import annotations

import pathlib
from typing import Literal

from pydantic import BaseModel, Field


class PlotConfig(BaseModel, frozen=True):
    """Rendering options for PyVista slice-error visualization."""

    slice_mode: Literal["xy", "yz", "xz"]
    slice_origin: list[float]

    def get_slice_normal(self) -> list[float]:
        match self.slice_mode:
            case "xy":
                return [0.0, 0.0, 1.0]
            case "yz":
                return [1.0, 0.0, 0.0]
            case "xz":
                return [0.0, 1.0, 0.0]
            case _:
                raise ValueError(f"Invalid slice mode: {self.slice_mode}")


class OpenFOAMRunConfig(BaseModel, frozen=True):
    """Optional ``bash`` steps under an OpenFOAM case directory."""

    enabled: bool = False
    allrun_cmd: str = "./Allrun"


class GridfoamRunConfig(BaseModel, frozen=True):
    """Optional ``uv run python <script>`` for a gridfoam driver script."""

    enabled: bool = False
    script: str = "run.py"


class CompareConfig(BaseModel, frozen=True):
    """
    Full comparison parameters.

    Parameters
    ----------
    reference_vtk
        Glob or path (relative to repo root) to ``foamToVTK`` output.
        When multiple paths match, the last sorted match is used.
    compare_vtu
        gridfoam ``.vtu`` file (cell fields from :mod:`gridfoam.io.vtu`).
    """

    experiment_name: str
    case_dir: pathlib.Path
    field_name: str
    field_kind: Literal["scalar", "vector"]
    plot: PlotConfig
    openfoam: OpenFOAMRunConfig = Field(default_factory=OpenFOAMRunConfig)
    gridfoam: GridfoamRunConfig = Field(default_factory=GridfoamRunConfig)
