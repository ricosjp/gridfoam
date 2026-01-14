"""Tests for EquationMeta."""

import pytest
import torch

from gridfoam.DNA.ASTNodes._interface import IASTNode
from gridfoam.DNA.enum import BoundaryConditionType, FieldLayout, FieldRole
from gridfoam.DNA.meta.boundary_condition import BoundaryConditionMeta
from gridfoam.DNA.meta.equation import EquationMeta
from gridfoam.DNA.meta.field import FieldMeta


def test_equation_meta_basic():
    """Test basic EquationMeta creation."""
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
    ast_root = target_field  # Simple AST node

    eq = EquationMeta(
        name="momentum",
        target_field=target_field,
        boundary_conditions=[bc],
        ast_root=ast_root,
    )
    assert eq.name == "momentum"
    assert eq.target_field == target_field
    assert eq.boundary_conditions == [bc]
    assert eq.ast_root == ast_root


def test_equation_meta_non_state_field_error():
    """Test EquationMeta raises error for non-state target field."""
    target_field = FieldMeta(
        name="nu",
        label="Viscosity",
        role=FieldRole.AUXILIARY,
    )
    bc = BoundaryConditionMeta(
        name="inlet",
        target_field=target_field,
        type=BoundaryConditionType.DIRICHLET,
        value=torch.tensor([1.0]),
    )
    ast_root = target_field

    with pytest.raises(ValueError, match="must be a state field"):
        EquationMeta(
            name="test",
            target_field=target_field,
            boundary_conditions=[bc],
            ast_root=ast_root,
        )
