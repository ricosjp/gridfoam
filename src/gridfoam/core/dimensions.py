"""Physical-dimension metadata helpers built on phlower-tensor."""

from __future__ import annotations

from phlower_tensor import PhysicalDimensions, phlower_dimension_tensor

from gridfoam.core.name import parse_field_name

DimensionLike = PhysicalDimensions | dict[str, float | int] | None

DIMLESS = PhysicalDimensions({})
DIM_LENGTH = PhysicalDimensions({"L": 1})
DIM_AREA = PhysicalDimensions({"L": 2})
DIM_VOLUME = PhysicalDimensions({"L": 3})
DIM_TIME = PhysicalDimensions({"T": 1})
DIM_VELOCITY = PhysicalDimensions({"L": 1, "T": -1})
DIM_KIN_PRESSURE = PhysicalDimensions({"L": 2, "T": -2})
DIM_VOL_FLUX = PhysicalDimensions({"L": 3, "T": -1})
DIM_NU = PhysicalDimensions({"L": 2, "T": -1})
DIM_RAU = PhysicalDimensions({"T": 1})
DIM_VELOCITY_POTENTIAL = PhysicalDimensions({"L": 2, "T": -1})

_FIELD_DEFAULTS: dict[str, PhysicalDimensions] = {
    "U": DIM_VELOCITY,
    "p": DIM_KIN_PRESSURE,
    "phi": DIM_VOL_FLUX,
    "rAU": DIM_RAU,
    "HbyA": DIM_VELOCITY,
    "nu_t": DIM_NU,
}

_DERIVED_PREFIXES = ("grad(", "div(", "snGrad(")


class DimensionMismatchError(ValueError):
    """Raised when two tracked physical dimensions are incompatible."""


def to_dimensions(value: DimensionLike) -> PhysicalDimensions | None:
    """
    Normalize a dimension specification to ``PhysicalDimensions``.

    Parameters
    ----------
    value : PhysicalDimensions or dict[str, float or int] or None
        Dimension metadata, or ``None`` for untracked dimensions.

    Returns
    -------
    PhysicalDimensions or None
        Normalized dimension metadata, or ``None`` when tracking is
        disabled.
    """
    if value is None:
        return None
    if isinstance(value, PhysicalDimensions):
        return value
    return PhysicalDimensions({key: float(val) for key, val in value.items()})


def resolve_field_dimension(
    name: str,
    explicit: DimensionLike = None,
    config: DimensionLike = None,
) -> PhysicalDimensions | None:
    """
    Resolve field dimension metadata from explicit, config, and defaults.

    Parameters
    ----------
    name : str
        Field name, optionally with a phase suffix.
    explicit : PhysicalDimensions or dict[str, float or int] or None, optional
        Dimension passed by the caller. Takes the highest precedence.
    config : PhysicalDimensions or dict[str, float or int] or None, optional
        Dimension declared in the simulator condition config. Used when
        ``explicit`` is ``None``.

    Returns
    -------
    PhysicalDimensions or None
        Resolved dimension metadata, or ``None`` when no source applies.
    """
    if explicit is not None:
        return to_dimensions(explicit)
    if config is not None:
        return to_dimensions(config)
    return default_field_dimension(name)


def default_field_dimension(name: str) -> PhysicalDimensions | None:
    """
    Return the default kinematic dimension for a known primary field name.

    Derived and interpolated fields return ``None`` so that callers set
    dimensions explicitly from the dimensions of their operands.

    Parameters
    ----------
    name : str
        Field name, optionally with a phase suffix.

    Returns
    -------
    PhysicalDimensions or None
        Default dimension for the field, or ``None`` when unknown.
    """
    base = parse_field_name(name).base
    if base.startswith(_DERIVED_PREFIXES):
        return None
    if base.endswith("_f"):
        return None
    return _FIELD_DEFAULTS.get(base)


def assert_compatible(
    left: PhysicalDimensions | None,
    right: PhysicalDimensions | None,
    context: str = "",
) -> None:
    """
    Assert that two tracked dimensions are equal.

    The check is skipped when either side is ``None``, which means the
    corresponding quantity does not track dimensions.

    Parameters
    ----------
    left : PhysicalDimensions or None
        First dimension to compare.
    right : PhysicalDimensions or None
        Second dimension to compare.
    context : str, optional
        Description prepended to the error message.

    Raises
    ------
    DimensionMismatchError
        If both dimensions are tracked and differ.
    """
    if left is None or right is None:
        return
    if left != right:
        prefix = f"{context}: " if context else ""
        raise DimensionMismatchError(
            f"{prefix}incompatible dimensions {left.to_dict()} and "
            f"{right.to_dict()}"
        )


def dim_mul(
    left: PhysicalDimensions | None,
    right: PhysicalDimensions | None,
) -> PhysicalDimensions | None:
    """
    Multiply two dimensions by adding their exponents.

    Parameters
    ----------
    left : PhysicalDimensions or None
        Left operand.
    right : PhysicalDimensions or None
        Right operand.

    Returns
    -------
    PhysicalDimensions or None
        Product dimension, or ``None`` if either operand is untracked.
    """
    if left is None or right is None:
        return None
    result = phlower_dimension_tensor(left) * phlower_dimension_tensor(right)
    return result.to_physics_dimension()


def dim_div(
    left: PhysicalDimensions | None,
    right: PhysicalDimensions | None,
) -> PhysicalDimensions | None:
    """
    Divide two dimensions by subtracting their exponents.

    Parameters
    ----------
    left : PhysicalDimensions or None
        Numerator dimension.
    right : PhysicalDimensions or None
        Denominator dimension.

    Returns
    -------
    PhysicalDimensions or None
        Quotient dimension, or ``None`` if either operand is untracked.
    """
    if left is None or right is None:
        return None
    result = phlower_dimension_tensor(left) / phlower_dimension_tensor(right)
    return result.to_physics_dimension()


def dim_pow(
    dimension: PhysicalDimensions | None,
    exponent: float,
) -> PhysicalDimensions | None:
    """
    Raise a dimension to a power by scaling its exponents.

    Parameters
    ----------
    dimension : PhysicalDimensions or None
        Base dimension.
    exponent : float
        Power to raise the dimension to.

    Returns
    -------
    PhysicalDimensions or None
        Resulting dimension, or ``None`` if the base is untracked.
    """
    if dimension is None:
        return None
    result = phlower_dimension_tensor(dimension) ** exponent
    return result.to_physics_dimension()


def dimension_config(dimension: PhysicalDimensions) -> dict[str, float]:
    """
    Return a compact dimension dict suitable for YAML / config models.

    Parameters
    ----------
    dimension : PhysicalDimensions
        Dimension metadata to serialize.

    Returns
    -------
    dict[str, float]
        Non-zero SI base exponents keyed by symbol.
    """
    return {
        key: float(value)
        for key, value in dimension.to_dict().items()
        if value != 0.0
    }
