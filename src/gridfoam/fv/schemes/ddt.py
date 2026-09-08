"""Shared Euler/backward time weights for the matrix and flux correction."""

import math
from collections.abc import Callable

from gridfoam.core.field import CellField
from gridfoam.fv.schemes.selection import search_ddt_scheme
from gridfoam.meta.enums import DdtScheme, FieldRole

DdtSchemeFunc = Callable[[CellField], tuple[float, float, float]]


def euler(_field: CellField) -> tuple[float, float, float]:
    """Return first-order Euler weights ``(1, 1, 0)``."""
    return 1.0, 1.0, 0.0


def backward(field: CellField) -> tuple[float, float, float]:
    """
    Return BDF2 weights, falling back to Euler without older history.

    OpenCFD v2606 backward coefficients include unequal consecutive time
    intervals. Missing older history uses an Euler startup step.
    """
    if field.role != FieldRole.TRANSIENT:
        raise ValueError("backward ddt requires a TRANSIENT cell field")
    if field.older_data is None:
        return 1.0, 1.0, 0.0
    dt = field.grid.dt
    dt0 = field.previous_dt
    if dt0 is None or not math.isfinite(dt0) or dt0 <= 0:
        raise ValueError(
            "backward ddt requires a finite positive previous time step"
        )
    ratio = dt / dt0
    a = 1.0 + dt / (dt + dt0)
    c = ratio * dt / (dt + dt0)
    return a, a + c, c


DDT_SCHEMES: dict[DdtScheme, DdtSchemeFunc] = {
    DdtScheme.EULER: euler,
    DdtScheme.BACKWARD: backward,
}


def get_ddt_scheme(scheme: DdtScheme) -> DdtSchemeFunc:
    """Return the time-scheme coefficient function for the enum."""
    return DDT_SCHEMES[scheme]


def ddt_coefficients(field: CellField) -> tuple[float, float, float]:
    """Return (current, old, older) in (a*x - b*x_old + c*x_older)/dt."""
    dt = field.grid.dt
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Time step must be finite and positive")
    scheme = search_ddt_scheme(field.grid.sim_config, field)
    return get_ddt_scheme(scheme)(field)
