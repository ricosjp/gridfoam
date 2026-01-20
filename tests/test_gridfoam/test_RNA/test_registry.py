"""Tests for SimulationMetaRegistry."""

import pytest
import torch

from gridfoam.DNA.config import fvSchemesConfig
from gridfoam.DNA.enum import (
    BoundaryConditionType,
    FieldLayout,
    FieldRole,
)
from gridfoam.DNA.meta.boundary_condition import BoundaryConditionMeta
from gridfoam.DNA.meta.equation import EquationMeta
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.ddt._choice import FVMDdtSchemeChoice
from gridfoam.DNA.scheme.fvm.div._choice import FVMDivSchemeChoice
from gridfoam.DNA.scheme.fvm.grad._choice import FVMGradSchemeChoice
from gridfoam.DNA.scheme.fvm.laplacian._choice import FVMLaplacianSchemeChoice
from gridfoam.RNA.registry import SimulationMetaRegistry


def test_registry_register_field():
    """Test register_field method."""
    reg = SimulationMetaRegistry()
    field = FieldMeta(name="test", label="Test Field")
    reg.register_field(field)
    assert "test" in reg.fields
    assert reg.fields["test"] == field


def test_registry_register_field_duplicate():
    """Test register_field raises error for duplicate."""
    reg = SimulationMetaRegistry()
    field = FieldMeta(name="test", label="Test Field")
    reg.register_field(field)
    with pytest.raises(ValueError, match="already registered"):
        reg.register_field(field)


def test_registry_get_field():
    """Test get_field method."""
    reg = SimulationMetaRegistry()
    field = FieldMeta(name="test", label="Test Field")
    reg.register_field(field)
    retrieved = reg.get_field("test")
    assert retrieved == field


def test_registry_get_field_not_found():
    """Test get_field raises error for non-existent field."""
    reg = SimulationMetaRegistry()
    with pytest.raises(ValueError, match="not registered"):
        reg.get_field("nonexistent")


def test_registry_register_equation():
    """Test register_equation method."""
    reg = SimulationMetaRegistry()
    target_field = FieldMeta(
        name="U",
        label="Velocity",
        role=FieldRole.STATE,
    )
    bc = BoundaryConditionMeta(
        name="inlet",
        target_field=target_field,
        type=BoundaryConditionType.DIRICHLET,
        value=torch.tensor([1.0, 0.0, 0.0]),
    )
    eq = EquationMeta(
        name="momentum",
        target_field=target_field,
        boundary_conditions=[bc],
        ast_root=target_field,
    )
    reg.register_equation(eq)
    assert "momentum" in reg.equations
    assert reg.equations["momentum"] == eq


def test_registry_register_equation_duplicate():
    """Test register_equation raises error for duplicate."""
    reg = SimulationMetaRegistry()
    target_field = FieldMeta(
        name="U",
        label="Velocity",
        role=FieldRole.STATE,
    )
    bc = BoundaryConditionMeta(
        name="inlet",
        target_field=target_field,
        type=BoundaryConditionType.DIRICHLET,
        value=torch.tensor([1.0, 0.0, 0.0]),
    )
    eq = EquationMeta(
        name="momentum",
        target_field=target_field,
        boundary_conditions=[bc],
        ast_root=target_field,
    )
    reg.register_equation(eq)
    with pytest.raises(ValueError, match="already registered"):
        reg.register_equation(eq)


def test_registry_register_scheme():
    """Test register_scheme method."""
    reg = SimulationMetaRegistry()
    schemes = fvSchemesConfig(
        ddtSchemes={"default": FVMDdtSchemeChoice.EULER},
        divSchemes={"default": FVMDivSchemeChoice.UPWIND},
        laplacianSchemes={"default": FVMLaplacianSchemeChoice.LINEAR},
        gradSchemes={"default": FVMGradSchemeChoice.LINEAR},
    )
    reg.register_scheme(schemes)
    assert "default" in reg.ddt_scheme_configs
    assert "default" in reg.div_scheme_configs
    assert "default" in reg.laplacian_scheme_configs
    assert "default" in reg.grad_scheme_configs


def test_registry_get_ddt_operator():
    """Test get_ddt_operator method."""
    reg = SimulationMetaRegistry()
    schemes = fvSchemesConfig(
        ddtSchemes={"default": FVMDdtSchemeChoice.EULER},
    )
    reg.register_scheme(schemes)
    field = FieldMeta(
        name="test",
        label="Test",
        layout=FieldLayout.CELL,
    )
    operator = reg.get_ddt_operator("default", field)
    assert operator is not None


def test_registry_get_ddt_operator_not_found():
    """Test get_ddt_operator raises error for non-existent scheme."""
    reg = SimulationMetaRegistry()
    field = FieldMeta(
        name="test",
        label="Test",
        layout=FieldLayout.CELL,
    )
    with pytest.raises(ValueError, match="not registered"):
        reg.get_ddt_operator("nonexistent", field)
