"""Shared Euler/backward time weights for the matrix and flux correction."""

import math

from gridfoam.core.field import CellField
from gridfoam.meta.enums import DdtScheme, FieldRole


def ddt_coefficients(field: CellField) -> tuple[float, float, float]:
    """Return (current, old, older) in (a*x - b*x_old + c*x_older)/dt.

    OpenCFD v2606 backward coefficients include unequal consecutive time
    intervals. Missing older history uses an Euler startup step.
    """
    dt = field.grid.dt
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Time step must be finite and positive")
    schemes = field.grid.sim_config.fvSchemes.ddtSchemes or {}
    scheme = schemes.get(
        f"ddt({field.name})", schemes.get("default", DdtScheme.EULER)
    )
    if scheme == DdtScheme.EULER:
        return 1.0, 1.0, 0.0
    if field.role != FieldRole.TRANSIENT:
        raise ValueError("backward ddt requires a TRANSIENT cell field")
    if field.older_data is None:
        return 1.0, 1.0, 0.0
    dt0 = field.previous_dt
    if dt0 is None or not math.isfinite(dt0) or dt0 <= 0:
        raise ValueError(
            "backward ddt requires a finite positive previous time step"
        )
    ratio = dt / dt0
    a = 1.0 + dt / (dt + dt0)
    c = ratio * dt / (dt + dt0)
    return a, a + c, c
