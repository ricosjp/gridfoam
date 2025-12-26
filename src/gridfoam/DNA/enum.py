from __future__ import annotations

from enum import Enum, auto


class FieldRole(Enum):
    """
    Role of the variable.
    - STATE: Time-dependent field variables that are expressed in equations.
        e.g. velocity, temperature, pressure, etc.
    - AUXILIARY: Field variables that are required for computational model.
        e.g. viscosity, k-epsilon in turbulence models, etc.
    - DIAGNOSTIC: Field variables that are used for monitoring.
        Therefore, they are calculated separately from the state equations.
        e.g. vorticity, divergence, energy, CFL, etc.
    """

    STATE = auto()
    AUXILIARY = auto()
    DIAGNOSTIC = auto()


class FieldLayout(Enum):
    """
    Layout of the field.
    - CELL: Cell-centered variable.
    - FACE: Face-centered variable
    """

    CELL = auto()
    FACE = auto()


class Precision(Enum):
    FLOAT32 = auto()
    FLOAT64 = auto()


class Device(Enum):
    CPU = auto()
    CUDA = auto()


class Axis(Enum):
    X = auto()
    Y = auto()
    Z = auto()


class NormType(Enum):
    """
    Type of the norm.
    - L_inf: Infinity norm.
    - L_2: 2-norm.
    """

    L_inf = auto()
    L_2 = auto()


class TimeLevel(Enum):
    """
    Time level of the field.
    - NEW: New time level.
    - OLD: Old time level.
    - OLD_OLD: Old old time level.
    """

    CUR = auto()
    OLD = auto()
    OLD_OLD = auto()


class BoundaryConditionType(Enum):
    """
    Type of the boundary condition.
    - DIRICHLET: Dirichlet boundary condition.
    - NEUMANN: Neumann boundary condition.
    """

    DIRICHLET = auto()
    NEUMANN = auto()


class OperatorType(Enum):
    """
    Type of the operator.
    - LAPLACIAN: Laplacian operator.
    - DIV: Divergence operator.
    - DDT: Time derivative operator.
    - GRAD: Gradient operator.
    """

    DDT = "ddt"
    DIV = "div"
    GRAD = "grad"
    LAPLACIAN = "laplacian"
