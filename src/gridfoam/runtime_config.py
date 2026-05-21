from __future__ import annotations

import os

RUNTIME_TYPE_CHECKS_ENV = "GRIDFOAM_RUNTIME_TYPE_CHECKS"

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}


def runtime_type_checks_enabled(*, default: bool = True) -> bool:
    value = os.environ.get(RUNTIME_TYPE_CHECKS_ENV)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False

    msg = (
        f"{RUNTIME_TYPE_CHECKS_ENV} must be one of "
        f"{sorted(_TRUE_VALUES | _FALSE_VALUES)}, got {value!r}"
    )
    raise ValueError(msg)
