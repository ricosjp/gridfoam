import pathlib
import re
from collections.abc import Iterable
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from gridfoam.meta.enums import (
    BoundaryConditionType,
    DdtScheme,
    DeviceType,
    DivScheme,
    DomainBoundaryPatch,
    GradScheme,
    IbmType,
    LaplacianScheme,
    NormType,
    PrecisionType,
    PreconditionerType,
    SolverType,
)


class DomainConfig(BaseModel, frozen=True):
    lower: list[float]
    """
    lower : list[float, float, float]
        Lower bound of the domain.
    """
    upper: list[float]
    """
    upper : list[float, float, float]
        Upper bound of the domain.
    """


class RefinementRegionConfig(BaseModel, frozen=True):
    name: str | None = None
    """
    name : str | None
        Optional label for this local refinement region.
    """
    min: list[float]
    """
    min : list[float, float, float]
        Lower corner of the axis-aligned refinement box.
    """
    max: list[float]
    """
    max : list[float, float, float]
        Upper corner of the axis-aligned refinement box.
    """
    level: int
    """
    level : int
        Target octree refinement level for cells intersecting this box.
    """

    @field_validator("min", "max")
    @classmethod
    def validate_corner(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("refinement region corners must have 3 values")
        return value

    @field_validator("level")
    @classmethod
    def validate_level(cls, value: int) -> int:
        if value < 0:
            raise ValueError("refinement region level must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_bounds(self) -> "RefinementRegionConfig":
        if any(lo >= hi for lo, hi in zip(self.min, self.max, strict=True)):
            raise ValueError("refinement region min must be less than max")
        return self


class FluxelConfig(BaseModel, frozen=True):
    domain: DomainConfig
    """
    domain : DomainConfig
        Domain configuration.
    """
    root_resolution: list[int]
    """
    root_resolution : list[int, int, int]
        Resolution configuration.
        This block is treated as the octree root and refined recursively.
    """
    target_level: int
    """
    target_level : int
        The maximum octree refinement level around the input surface.
    """
    n_leaf_refinement: int
    """
    n_leaf_refinement : int
        The number of times that the uniform refinement is applied
        to the final mesh.
        n_leaf_refinement=3 will generate 8x8x8 micro-cells per octree leaf.
    """
    refinement_regions: list[RefinementRegionConfig] = Field(
        default_factory=list
    )
    """
    refinement_regions : list[RefinementRegionConfig]
        Additional axis-aligned boxes to refine after surface-based AMR and
        before 2:1 balancing and final uniform leaf refinement.
    """
    mesh_path: pathlib.Path | None = None
    """
    mesh_path : pathlib.Path | None
        Path to the mesh file. (STL or OBJ)
        None will generate a mesh without immersed boundary.
    """
    ibm_type: IbmType
    """
    ibm_type : IbmType
        Type of the immersed boundary method.
        Currently supported methods:
            - AXIS_PROJECTED: Axis projected method.
    """


class OutputConfig(BaseModel, frozen=True):
    output_dir: pathlib.Path
    """
    output_dir : pathlib.Path
        Output directory for the grid.
    """
    base_name: str
    """
    base_name : str
        Base name for the output files.
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
    output: OutputConfig
    """
    output : OutputConfig
        Output configuration.
    """
    precision: PrecisionType
    """
    precision : PrecisionType
        Precision type for the simulation.
        Currently supported precisions:
            - FLOAT32: 32-bit floating point.
            - FLOAT64: 64-bit floating point.
    """


class fvSchemesConfig(BaseModel, frozen=True):
    ddtSchemes: dict[str, DdtScheme] | None = None
    """
    ddtSchemes : dict[str, DdtScheme] | None
        Scheme for the ddt.
        Currently supported schemes:
            - EULER: Euler scheme.
    """
    gradSchemes: dict[str, GradScheme] | None = None
    """
    gradSchemes : dict[str, gradScheme] | None
        Scheme for the grad.
        Currently supported schemes:
            - LINEAR: Linear scheme.
    """
    divSchemes: dict[str, DivScheme] | None = None
    """
    divSchemes : dict[str, DivScheme] | None
        Scheme for the div.
        Currently supported schemes:
            - UPWIND: Upwind scheme.
            - LINEAR: Linear scheme.
    """
    laplacianSchemes: dict[str, LaplacianScheme] | None = None
    """
    laplacianSchemes : dict[str, LaplacianScheme] | None
        Scheme for the laplacian.
        Currently supported schemes:
            - LINEAR: Linear scheme.
    """

    @field_validator(
        "ddtSchemes",
        "divSchemes",
        "laplacianSchemes",
        "gradSchemes",
        mode="before",
    )
    @classmethod
    def regularize_keys(
        cls, d: dict[str, Enum] | None
    ) -> dict[str, Enum] | None:
        if d is None:
            return None
        return {re.sub(r"\s*,\s*", ", ", k): v for k, v in d.items()}


class SolverConfig(BaseModel, frozen=True):
    method: SolverType
    """
    method : SolverType
        Type of the solver.
        Currently supported solvers:
            - CG: Conjugate Gradient method.
            - BiCGSTAB: Bi-Conjugate Gradient Stabilized method.
    """
    preconditioner: PreconditionerType = Field(default=PreconditionerType.NONE)
    """
    preconditioner : PreconditionerType, default=PreconditionerType.NONE
        Preconditioner for the solver.
        Currently supported preconditioners:
            - NONE: Identity preconditioner (no-op, ``M = I``).
            - JACOBI: Jacobi preconditioner.
    """
    tolerance: float = Field(default=1e-6)
    """
    tolerance : float, default=1e-6
        Tolerance for the solver.
    """
    rel_tolerance: float = Field(default=1e-6)
    """
    rel_tolerance : float, default=1e-6
        Relative tolerance for the solver.
    """
    max_iter: int = Field(default=1000)
    """
    max_iter : int, default=1000
        Maximum number of iterations for the solver.
    """
    norm_type: NormType = Field(default=NormType.L_2)
    """
    norm_type : NormType, default=NormType.L_2
        Type of the norm for the solver.
    """
    max_restart: int = Field(default=5)
    """
    max_restart : int, default=5
        Maximum restart count for the solver.
    """
    log_interval: int = Field(default=25)
    """
    log_interval : int, default=25
        Iteration interval for DEBUG residual logging.
    """


class fvSolutionConfig(BaseModel, frozen=True):
    solvers: dict[str, SolverConfig]
    """
    solvers : dict[str, SolverConfig]
        Solvers configuration.
        The key is the target equation name to be solved by the solver.
        The value is the solver configuration.
    """


class BoundaryConditionConfig(BaseModel, frozen=True):
    name: str
    """
    name : str
        Name of the boundary condition. This is just for labeling
        and does not affect the behavior of the boundary condition.
    """
    type: BoundaryConditionType
    """
    Boundary condition type.
    """
    patches: list[str | DomainBoundaryPatch]
    """
    Target patch names for this boundary condition.
    """
    value: list[float] | None = None
    """
    Numeric payload used by value-based boundary conditions.
    """
    phi_builtin_key: str | None = None
    HbyA_builtin_key: str | None = None
    rAU_builtin_key: str | None = None

    @field_validator("patches", mode="before")
    @classmethod
    def regularize_patches(
        cls, patches: object
    ) -> list[str | DomainBoundaryPatch]:
        if not isinstance(patches, Iterable):
            raise TypeError("patches must be an iterable of patch names")
        _DOMAIN_BOUNDARY_PATCH_BY_VALUE: dict[str, DomainBoundaryPatch] = {
            m.value: m for m in DomainBoundaryPatch
        }
        out: list[str | DomainBoundaryPatch] = []
        for p in patches:
            if not isinstance(p, str):
                raise TypeError("patch names must be strings")
            reserved = _DOMAIN_BOUNDARY_PATCH_BY_VALUE.get(p)
            out.append(reserved if reserved is not None else p)
        return out


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
    boundaryConditions: dict[str, list[BoundaryConditionConfig]] | None = None
    """
    Boundary condition configuration grouped by field name.
    """
    device: DeviceType
    """
    device : DeviceType
        Device type for the simulation.
        Currently supported devices:
            - CPU: CPU.
            - CUDA: CUDA.
    """


class GridfoamConfig(BaseModel, frozen=True):
    simulator: SimulatorConfig
    """
    simulator : SimulatorConfig
        Simulator configuration.
    """

    fluxel: FluxelConfig
    """
    fluxel : FluxelConfig
        Fluxel configuration.
    """
