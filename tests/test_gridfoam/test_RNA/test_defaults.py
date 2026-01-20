"""Tests for defaults module."""

from gridfoam.RNA.defaults import default_registry
from gridfoam.RNA.registry import SimulationMetaRegistry


def test_default_registry():
    """Test default_registry returns SimulationMetaRegistry."""
    reg = default_registry()
    assert isinstance(reg, SimulationMetaRegistry)


def test_default_registry_has_builtin_fields():
    """Test default_registry has builtin fields registered."""
    reg = default_registry()
    # Check that builtin fields are registered
    assert "p" in reg.fields
    assert "U" in reg.fields
    assert "phi" in reg.fields
