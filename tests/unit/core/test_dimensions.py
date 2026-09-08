"""
Physical dimensions follow kinematic presets, explicit overrides, and field
compatibility.
"""

from __future__ import annotations

import pytest
from tests.helpers.grids import refined_grid

from gridfoam.core.dimensions import (
    DIM_KIN_PRESSURE,
    DIM_LENGTH,
    DIM_VELOCITY,
    DIM_VOL_FLUX,
    DimensionMismatchError,
    assert_compatible,
    default_field_dimension,
    dim_div,
    dim_mul,
    resolve_field_dimension,
    to_dimensions,
)
from gridfoam.core.field import CellField, get_or_create_cellfield
from gridfoam.fv import fvc
from gridfoam.meta.enums import FieldRole


def test_preset_dimensions_match_kinematic_conventions() -> None:
    """
    U, p, and phi use kinematic dimensions; derived field names have no
    preset.
    """
    assert default_field_dimension("U") == DIM_VELOCITY
    assert default_field_dimension("p") == DIM_KIN_PRESSURE
    assert default_field_dimension("phi") == DIM_VOL_FLUX
    assert default_field_dimension("grad(p)") is None


def test_dim_mul_and_dim_div_propagate_exponents() -> None:
    """
    Length/time yields velocity, and velocity times area yields volumetric
    flux.
    """
    velocity = dim_div(DIM_LENGTH, to_dimensions({"T": 1}))
    assert velocity == DIM_VELOCITY

    flux = dim_mul(DIM_VELOCITY, to_dimensions({"L": 2}))
    assert flux == DIM_VOL_FLUX


def test_assert_compatible_skips_when_tracking_disabled() -> None:
    """Either missing dimension disables compatibility checking."""
    assert_compatible(None, DIM_VELOCITY, "ignored")
    assert_compatible(DIM_VELOCITY, None, "ignored")


def test_assert_compatible_raises_on_mismatch() -> None:
    """
    Velocity and kinematic pressure dimensions cannot be treated as
    compatible.
    """
    with pytest.raises(DimensionMismatchError):
        assert_compatible(
            DIM_VELOCITY, DIM_KIN_PRESSURE, "velocity vs pressure"
        )


def test_resolve_field_dimension_prefers_explicit_over_default() -> None:
    """
    An explicit dimension overrides both configuration and the field-name
    preset.
    """
    resolved = resolve_field_dimension(
        "U",
        explicit={"Theta": 1},
        config={"L": 1, "T": -1},
    )
    assert resolved == to_dimensions({"Theta": 1})


def test_grad_pressure_dimension_is_acceleration_like() -> None:
    """
    The pressure gradient divides kinematic pressure dimensions by length.
    """
    grid = refined_grid()
    p = CellField(
        grid,
        "p",
        FieldRole.LOCAL,
        component_shape=(),
        dimension=DIM_KIN_PRESSURE,
    )
    grad_p = fvc.grad(p)
    expected = dim_div(DIM_KIN_PRESSURE, DIM_LENGTH)
    assert grad_p.dimension == expected
    assert grad_p.dimension == to_dimensions({"L": 1, "T": -2})


def test_get_or_create_cellfield_checks_existing_dimension() -> None:
    """
    Field reuse preserves identity but rejects a conflicting physical
    dimension.
    """
    grid = refined_grid()
    first = get_or_create_cellfield(
        grid, "U", FieldRole.LOCAL, (3,), dimension=DIM_VELOCITY
    )
    second = get_or_create_cellfield(
        grid, "U", FieldRole.LOCAL, (3,), dimension=DIM_VELOCITY
    )
    assert first is second

    with pytest.raises(DimensionMismatchError):
        get_or_create_cellfield(
            grid, "U", FieldRole.LOCAL, (3,), dimension=DIM_KIN_PRESSURE
        )
