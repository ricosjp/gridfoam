"""Tests for ctx_for_cube_operation module."""

import numpy as np
import pytest
import torch

from gridfoam.DNA.ctx_for_cube_operation import CtxForCubeOperation
from gridfoam.DNA.meta.boundary_condition import BoundaryConditionMeta


def test_ctx_for_cube_operation_import():
    """Test that CtxForCubeOperation can be imported."""
    from gridfoam.DNA.ctx_for_cube_operation import CtxForCubeOperation
    assert CtxForCubeOperation is not None


def test_ctx_for_cube_operation_init():
    """Test CtxForCubeOperation initialization."""
    depth = 0
    bounds = np.array([0, 0, 0], dtype=np.uint64)
    dt = 0.1
    dx = torch.tensor([1.0, 1.0, 1.0])
    vertices = torch.zeros((8, 3))
    bcs = []
    
    ctx = CtxForCubeOperation(
        depth=depth,
        bounds=bounds,
        dt=dt,
        dx=dx,
        vertices=vertices,
        bcs=bcs,
    )
    assert ctx.depth == depth
    assert np.array_equal(ctx.bounds, bounds)
    assert ctx.dt == dt
    assert torch.equal(ctx.dx, dx)
    assert torch.equal(ctx.vertices, vertices)
    assert ctx.bcs == bcs


def test_ctx_for_cube_operation_with_bcs():
    """Test CtxForCubeOperation with boundary conditions."""
    from gridfoam.DNA.enum import BoundaryConditionType, FieldLayout, FieldRole
    from gridfoam.DNA.meta.field import FieldMeta
    
    depth = 0
    bounds = np.array([0, 0, 0], dtype=np.uint64)
    dt = 0.1
    dx = torch.tensor([1.0, 1.0, 1.0])
    vertices = torch.zeros((8, 3))
    
    field = FieldMeta(
        name="test",
        label="Test",
        layout=FieldLayout.CELL,
        role=FieldRole.STATE,
    )
    bc = BoundaryConditionMeta(
        name="inlet",
        target_field=field,
        type=BoundaryConditionType.DIRICHLET,
        value=torch.tensor([1.0]),
    )
    bcs = [bc]
    
    ctx = CtxForCubeOperation(
        depth=depth,
        bounds=bounds,
        dt=dt,
        dx=dx,
        vertices=vertices,
        bcs=bcs,
    )
    assert len(ctx.bcs) == 1
    assert ctx.bcs[0] == bc
