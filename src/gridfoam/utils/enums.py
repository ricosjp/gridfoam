from enum import Enum


class GridOutputMode(Enum):
    """
    Output mode for saving the grid.
    """

    CELL = "cell"
    CUBE = "cube"

class Axis(Enum):
    """
    Axis of the grid.
    """

    X = 0
    Y = 1
    Z = 2

class Direction(Enum):
    """
    Direction of the grid.
    """

    ZM = 4
    YM = 10
    XM = 12
    CENTER = 13
    XP = 14
    YP = 16
    ZP = 22


FACE_NEIGHBOR_MAP = {
    Direction.XM.value: (Axis.X, False), # -x
    Direction.XP.value: (Axis.X, True),  # +x
    Direction.YM.value: (Axis.Y, False), # -y
    Direction.YP.value: (Axis.Y, True),  # +y
    Direction.ZM.value: (Axis.Z, False), # -z
    Direction.ZP.value: (Axis.Z, True),  # +z
}


class TVDScheme(Enum):
    """
    TVD scheme for interpolation.
    """

    SUPERBEE = "superbee"
    MINMOD = "minmod"
    LIMITED_LINEAR = "limited_linear"
    VAN_LEER = "van_leer"
    VAN_ALBADA = "van_albada"
    UPWIND = "upwind"

class Namespace(Enum):
    """
    Namespace for the grid.
    """

    USER = "user"
    SOLVER = "solver"
    RHIE_CHOW = "rhie_chow"

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

    STATE = "state"
    AUXILIARY = "auxiliary"
    DIAGNOSTIC = "diagnostic"

class FieldLayout(Enum):
    """
    Layout of the field.
    - CELL: Cell-centered variable.
    - FACE: Face-centered variable
    - CUBE: Cube-centered variable.
    """

    CELL = "cell"
    FACE = "face"
    CUBE = "cube"
    # NODE = "node"

class ProlongateScheme(Enum):
    """
    Prolongate scheme for the field.
    """

    NEAREST = "nearest"
    TRILINEAR = "trilinear"


class RestrictScheme(Enum):
    """
    Restrict scheme for the field.
    """

    NEAREST = "nearest"
    TRILINEAR = "trilinear"

class FieldState(Enum):
    """
    State of the field.
    - CUR: Current state of the field.
    - OLD: Old state of the field.
    """

    CUR = "current"
    OLD = "old"

class SolverName(Enum):
    """
    Name of the solver.
    - CG: Conjugate Gradient method.
    - BICGSTAB: Bi-Conjugate Gradient Stabilized method.
    """

    CG = "CG"
    BICGSTAB = "BiCGSTAB"

class DiscretizationMode(Enum):
    """
    Mode of the discretization.
    - IMPLICIT: Implicit discretization.
    - EXPLICIT: Explicit discretization.
    """

    IMPLICIT = "implicit"
    EXPLICIT = "explicit"

class NormType(Enum):
    """
    Type of the norm.
    - L_inf: Infinity norm.
    - L_2: 2-norm.
    """

    L_inf = "L_inf"
    L_2 = "L_2"
