"""Tests for Rhie-Chow correction module."""

import pytest

from gridfoam.DNA.enum import FieldLayout
from gridfoam.DNA.meta.equation import EquationMeta
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.rhie_chow.correction import RhieChowCorrection


def test_rhie_chow_correction_import():
    """Test that RhieChowCorrection can be imported."""
    from gridfoam.DNA.scheme.rhie_chow.correction import RhieChowCorrection

    assert RhieChowCorrection is not None


def test_rhie_chow_correction_init():
    """Test RhieChowCorrection initialization."""
    U_field = FieldMeta(
        name="U",
        label="Velocity",
        layout=FieldLayout.CELL,
    )
    p_field = FieldMeta(
        name="p",
        label="Pressure",
        layout=FieldLayout.CELL,
    )
    momentum_eq = EquationMeta(
        name="momentum",
        target_field=U_field,
        boundary_conditions=[],
        ast_root=U_field,
    )

    correction = RhieChowCorrection(
        U_fm=U_field,
        p_fm=p_field,
        momentum_eq=momentum_eq,
    )
    assert correction._U_fm == U_field
    assert correction._p_fm == p_field
    assert correction._momentum_eq == momentum_eq


def test_rhie_chow_correction_init_errors():
    """Test RhieChowCorrection raises errors for invalid layouts."""
    U_field = FieldMeta(
        name="U",
        label="Velocity",
        layout=FieldLayout.FACE,  # Wrong layout
    )
    p_field = FieldMeta(
        name="p",
        label="Pressure",
        layout=FieldLayout.CELL,
    )
    momentum_eq = EquationMeta(
        name="momentum",
        target_field=U_field,
        boundary_conditions=[],
        ast_root=U_field,
    )

    with pytest.raises(AssertionError):
        RhieChowCorrection(
            U_fm=U_field,
            p_fm=p_field,
            momentum_eq=momentum_eq,
        )
