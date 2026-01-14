"""Tests for flux correction module."""

import pytest

from gridfoam.DNA.enum import FieldLayout
from gridfoam.DNA.meta.equation import EquationMeta
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.flux.correction import FluxCorrection


def test_flux_correction_import():
    """Test that FluxCorrection can be imported."""
    from gridfoam.DNA.scheme.flux.correction import FluxCorrection
    assert FluxCorrection is not None


def test_flux_correction_init():
    """Test FluxCorrection initialization."""
    phi_field = FieldMeta(
        name="phi",
        label="Flux",
        layout=FieldLayout.FACE,
    )
    U_field = FieldMeta(
        name="U",
        label="Velocity",
        layout=FieldLayout.CELL,
    )
    momentum_eq = EquationMeta(
        name="momentum",
        target_field=U_field,
        boundary_conditions=[],
        ast_root=U_field,
    )
    
    correction = FluxCorrection(
        phi_fm=phi_field,
        U_fm=U_field,
        momentum_eq=momentum_eq,
    )
    assert correction._phi_fm == phi_field
    assert correction._U_fm == U_field
    assert correction._momentum_eq == momentum_eq


def test_flux_correction_init_errors():
    """Test FluxCorrection raises errors for invalid layouts."""
    phi_field = FieldMeta(
        name="phi",
        label="Flux",
        layout=FieldLayout.CELL,  # Wrong layout
    )
    U_field = FieldMeta(
        name="U",
        label="Velocity",
        layout=FieldLayout.CELL,
    )
    momentum_eq = EquationMeta(
        name="momentum",
        target_field=U_field,
        boundary_conditions=[],
        ast_root=U_field,
    )
    
    with pytest.raises(AssertionError):
        FluxCorrection(
            phi_fm=phi_field,
            U_fm=U_field,
            momentum_eq=momentum_eq,
        )
