from __future__ import annotations

from dataclasses import dataclass

_PHASE_SEPARATOR = "."


@dataclass(frozen=True, slots=True)
class FieldNameParts:
    """
    Parsed ``(base, phase)`` components of a field name.

    Attributes
    ----------
    base : str
        Base field name without a phase suffix.
    phase : str or None
        Optional phase qualifier, or ``None`` for single-phase names.
    """

    base: str
    """Base field name without a phase suffix."""

    phase: str | None
    """Optional phase qualifier, or ``None`` for single-phase names."""


def make_field_name(base: str, *, phase: str | None = None) -> str:
    """
    Build a field name in OpenFOAM-style ``base`` or ``base.phase``
    format.

    Parameters
    ----------
    name
        Base field name.
    phase
        Optional phase qualifier for multiphase simulations.

    Returns
    -------
    str
        Field name.
    """
    if _PHASE_SEPARATOR in base:
        raise ValueError("field name must not contain '.'")
    if phase is not None:
        if _PHASE_SEPARATOR in phase:
            raise ValueError("phase must not contain '.'")
        return f"{base}{_PHASE_SEPARATOR}{phase}"
    return base


def parse_field_name(name: str) -> FieldNameParts:
    """
    Split a field key into its name and optional phase.

    Parameters
    ----------
    name
        Field name produced by :func:`make_field_name`.

    Returns
    -------
    FieldNameParts
        Parsed name and phase components.
    """
    if _PHASE_SEPARATOR not in name:
        return FieldNameParts(base=name, phase=None)
    base, phase = name.rsplit(_PHASE_SEPARATOR, 1)
    if not base or not phase:
        raise ValueError(f"invalid field name: {name!r}")
    return FieldNameParts(base=base, phase=phase)
