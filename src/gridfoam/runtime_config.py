from __future__ import annotations

import os

RUNTIME_TYPE_CHECKS_ENV = "GRIDFOAM_RUNTIME_TYPE_CHECKS"

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}


def runtime_type_checks_enabled(*, default: bool = True) -> bool:
    """
    Return whether jaxtyping runtime type checks are enabled.

    The environment variable ``GRIDFOAM_RUNTIME_TYPE_CHECKS`` overrides
    ``default`` when set. Accepted truthy values are ``1``, ``true``,
    ``yes``, ``on``, and ``enabled``; falsy values are ``0``, ``false``,
    ``no``, ``off``, and ``disabled``.

    Parameters
    ----------
    default : bool, optional
        Value used when the environment variable is unset. Default is
        ``True``.

    Returns
    -------
    bool
        Whether runtime type checks should be enabled.

    Raises
    ------
    ValueError
        If the environment variable is set to an unrecognized value.
    """
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
