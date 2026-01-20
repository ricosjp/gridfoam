"""Tests for builtin fields."""

from gridfoam.DNA.enum import FieldLayout, FieldRole
from gridfoam.RNA.builtins.builtin_fields import (
    builtin_p,
    builtin_phi,
    builtin_rAU,
    builtin_rho,
    builtin_T,
    builtin_U,
)


def test_builtin_U():
    """Test builtin_U function."""
    field = builtin_U()
    assert field.name == "U"
    assert field.label == "Velocity"
    assert field.role == FieldRole.STATE
    assert field.layout == FieldLayout.CELL
    assert field.components == 3
    assert field.unit == "m/s"


def test_builtin_p():
    """Test builtin_p function."""
    field = builtin_p()
    assert field.name == "p"
    assert field.label == "Pressure"
    assert field.role == FieldRole.STATE
    assert field.layout == FieldLayout.CELL
    assert field.components == 1
    assert field.unit == "Pa"


def test_builtin_phi():
    """Test builtin_phi function."""
    field = builtin_phi()
    assert field.name == "phi"
    assert field.label == "Flux"
    assert field.role == FieldRole.AUXILIARY
    assert field.layout == FieldLayout.FACE
    assert field.components == 1
    assert field.unit == "m^3/s"


def test_builtin_T():
    """Test builtin_T function."""
    field = builtin_T()
    assert field.name == "T"
    assert field.label == "Temperature"
    assert field.role == FieldRole.STATE
    assert field.layout == FieldLayout.CELL
    assert field.components == 1
    assert field.unit == "K"


def test_builtin_rho():
    """Test builtin_rho function."""
    field = builtin_rho()
    assert field.name == "rho"
    assert field.label == "Density"
    assert field.role == FieldRole.STATE
    assert field.layout == FieldLayout.CELL
    assert field.components == 1
    assert field.unit == "kg/m^3"


def test_builtin_rAU():
    """Test builtin_rAU function."""
    field = builtin_rAU()
    assert field.name == "rAU"
    assert field.role == FieldRole.AUXILIARY
    assert field.layout == FieldLayout.CELL
    assert field.components == 3
    assert field.unit == "1/s"
