from __future__ import annotations

from enum import StrEnum, auto

import torch
from fluxel import Direction


class IbmType(StrEnum):
    """Enum of immersed boundary method (IBM) types."""

    NONE = "none"
    AXIS_PROJECTED = "axis_projected"
    GHOST_CELL = "ghost_cell"


class DeviceType(StrEnum):
    """
    Type of the device.
    - CPU: CPU.
    - CUDA: CUDA.
    """

    CPU = auto()
    CUDA = auto()

    def to_torch_device(self: DeviceType) -> torch.device:
        """
        Map the device type to the torch device.
        """
        match self:
            case DeviceType.CPU:
                return torch.device("cpu")
            case DeviceType.CUDA:
                return torch.device("cuda")
            case _:
                raise ValueError(f"Unsupported device type: {self}")


class DomainBoundaryPatch(StrEnum):
    """
    Patch name of the domain boundary patch.
    """

    X_MINUS = auto()
    X_PLUS = auto()
    Y_MINUS = auto()
    Y_PLUS = auto()
    Z_MINUS = auto()
    Z_PLUS = auto()

    def to_direction(self: DomainBoundaryPatch) -> Direction:
        """
        Map the domain boundary patch name to the direction.
        """
        return DOMAIN_BOUNDARY_PATCH_TO_DIRECTION_MAP[self]


DOMAIN_BOUNDARY_PATCH_TO_DIRECTION_MAP = {
    DomainBoundaryPatch.X_MINUS: Direction.XMinus,
    DomainBoundaryPatch.X_PLUS: Direction.XPlus,
    DomainBoundaryPatch.Y_MINUS: Direction.YMinus,
    DomainBoundaryPatch.Y_PLUS: Direction.YPlus,
    DomainBoundaryPatch.Z_MINUS: Direction.ZMinus,
    DomainBoundaryPatch.Z_PLUS: Direction.ZPlus,
}


class FieldRole(StrEnum):
    """
    Tells the framework whether a field owns **time history**.

    Members
    -------
    TRANSIENT
        Primary evolving unknown: a **previous time level** is kept so
        transient terms can use both the new and the old value.
        Example: velocity, temperature, etc.
    LOCAL
        Supporting information (coefficients, intermediates, or other solved
        quantities). It does not use that history slot on this field.
        Example: pressure, flux, etc.
    MONITOR
        For monitoring or visualizing (plots, line probes, etc.).
        The next time step does not depend on storing or advancing this field.
    """

    TRANSIENT = auto()
    LOCAL = auto()
    MONITOR = auto()


class DdtScheme(StrEnum):
    """
    Type of the time derivative scheme.
    - EULER: Euler scheme.
    """

    EULER = auto()


class GradScheme(StrEnum):
    """
    Type of the gradient scheme.
    - LINEAR: Linear scheme.
    - LEASTSQUARE: Least-squares scheme.
    """

    LINEAR = auto()
    LEASTSQUARE = auto()


class DivScheme(StrEnum):
    """
    Type of the divergence scheme.
    - LINEAR: Linear scheme.
    - UPWIND: Upwind scheme.
    - VANLEER: Van Leer scheme.
    - MINMOD: Minmod scheme.
    - SUPERBEE: Superbee scheme.
    - MONOTONIZED_CENTRAL: Monotonized Central scheme. (MUSCL in OpenFOAM)
    - LIMITEDLINEAR: LimitedLinear scheme.
    - KOREN: Koren scheme.
    """

    UPWIND = auto()
    LINEAR = auto()
    VANLEER = auto()
    MINMOD = auto()
    SUPERBEE = auto()
    MONOTONIZED_CENTRAL = auto()
    LIMITEDLINEAR = auto()
    KOREN = auto()


class LaplacianScheme(StrEnum):
    """
    Type of the Laplacian scheme.
    - LINEAR: Linear scheme.
    """

    LINEAR = auto()


class SolverType(StrEnum):
    """
    Type of the solver.
    - CG: Conjugate Gradient method.
    - BiCGSTAB: Bi-Conjugate Gradient Stabilized method.
    - PyAMG: PyAMG method.
    """

    CG = auto()
    BiCGSTAB = auto()
    PyAMG = auto()


class NormType(StrEnum):
    """
    Type of the norm.
    - L_inf: Infinity norm.
    - L_2: 2-norm.
    """

    L_inf = auto()
    L_2 = auto()

    def to_norm_order(self: NormType) -> int | float:
        """
        Get the norm order for torch.
        """
        match self:
            case NormType.L_inf:
                return float("inf")
            case NormType.L_2:
                return 2
            case _:
                raise ValueError(f"Unsupported norm type: {self}")


class PreconditionerType(StrEnum):
    """
    Type of the linear-solver preconditioner.
    """

    NONE = auto()
    JACOBI = auto()


class PrecisionType(StrEnum):
    """
    Type of the precision.
    - FLOAT32: 32-bit floating point.
    - FLOAT64: 64-bit floating point.
    """

    FLOAT32 = auto()
    FLOAT64 = auto()

    def to_torch_dtype(self: PrecisionType) -> torch.dtype:
        """
        Map the precision type to the torch dtype.
        """
        match self:
            case PrecisionType.FLOAT32:
                return torch.float32
            case PrecisionType.FLOAT64:
                return torch.float64


class FaceSide(StrEnum):
    """
    Side of the face.
    This is used in Axis Projected Immersed Boundary Method (APIBM)
    for dual-sided treatment.
    - UPPER: Upper side (boundary face on owner to neighbour).
    - LOWER: Lower side (boundary face on neighbour to owner).
    """

    UPPER = auto()
    LOWER = auto()


class BoundaryConditionType(StrEnum):
    """
    Supported boundary condition types for YAML configuration.
    """

    DIRICHLET = auto()
    NEUMANN = auto()
    EMPTY = auto()
    SLIP = auto()
    INLET_OUTLET = auto()
    FIXED_FLUX_PRESSURE = auto()


class ForceCoordMode(StrEnum):
    """
    Mode of the force coordinate system.
    - DRAG_LIFT: Drag-lift coordinate system.
    - DRAG_PITCH: Drag-pitch coordinate system.
    """

    DRAG_LIFT = auto()
    DRAG_PITCH = auto()
