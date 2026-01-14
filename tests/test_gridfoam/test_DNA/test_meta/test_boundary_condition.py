"""Tests for BoundaryConditionMeta."""

import pytest
import torch

from gridfoam.DNA.enum import BoundaryConditionType, FieldLayout, FieldRole
from gridfoam.DNA.meta.boundary_condition import BoundaryConditionMeta
from gridfoam.DNA.meta.field import FieldMeta


def test_boundary_condition_meta_basic():
    """Test basic BoundaryConditionMeta creation."""
    field = FieldMeta(
        name="U",
        label="Velocity",
        role=FieldRole.STATE,
    )
    bc = BoundaryConditionMeta(
        name="inlet",
        target_field=field,
        type=BoundaryConditionType.DIRICHLET,
        value=torch.tensor([1.0, 0.0, 0.0]),
    )
    assert bc.name == "inlet"
    assert bc.target_field == field
    assert bc.type == BoundaryConditionType.DIRICHLET
    assert torch.equal(bc.value, torch.tensor([1.0, 0.0, 0.0]))
    assert bc.target_boundary_labels == []


def test_boundary_condition_meta_with_labels():
    """Test BoundaryConditionMeta with boundary labels."""
    field = FieldMeta(
        name="p",
        label="Pressure",
        role=FieldRole.STATE,
    )
    bc = BoundaryConditionMeta(
        name="outlet",
        target_field=field,
        type=BoundaryConditionType.NEUMANN,
        value=torch.tensor([0.0]),
        target_boundary_labels=["domainX+", "domainY+"],
    )
    assert bc.target_boundary_labels == ["domainX+", "domainY+"]
