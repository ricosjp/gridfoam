"""Tests for FVM div limited linear V operator."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from gridfoam.DNA._grid._grid import PyOctreeNode
from gridfoam.DNA.config import CubeConfig
from gridfoam.DNA.ctx_for_cube_operation import CtxForCubeOperation
from gridfoam.DNA.cubefield import CubeField
from gridfoam.DNA.enum import FieldLayout
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.div._limited_linearV import FVMDivLimitedLinearV


def test_fvm_div_limited_linear_v_init():
    """Test FVMDivLimitedLinearV initialization."""
    phi_field = FieldMeta(
        name="phi",
        label="Phi",
        layout=FieldLayout.FACE,
    )
    psi_field = FieldMeta(
        name="psi",
        label="Psi",
        layout=FieldLayout.CELL,
        components=3,  # Can handle multi-component
    )
    operator = FVMDivLimitedLinearV(phi_field, psi_field)
    assert operator._phi_fm == phi_field
    assert operator._psi_fm == psi_field
    assert operator._limited_linear is not None


def test_fvm_div_limited_linear_v_init_errors():
    """Test FVMDivLimitedLinearV raises errors for invalid layouts."""
    phi_field = FieldMeta(
        name="phi",
        label="Phi",
        layout=FieldLayout.CELL,  # Wrong layout
    )
    psi_field = FieldMeta(
        name="psi",
        label="Psi",
        layout=FieldLayout.CELL,
        components=3,
    )
    with pytest.raises(AssertionError):
        FVMDivLimitedLinearV(phi_field, psi_field)


def test_fvm_div_limited_linear_v_build_uniform_fields():
    """Test FVMDivLimitedLinearV build with uniform multi-component fields."""
    # Setup
    N = 8
    H = 2
    cube_config = CubeConfig(
        interior_width=N, halo_width=H, device=torch.device("cpu")
    )
    cube_field = CubeField(cube_config)

    phi_field = FieldMeta(
        name="phi",
        label="Phi",
        layout=FieldLayout.FACE,
        components=1,
        dtype=torch.float32,
    )
    psi_field = FieldMeta(
        name="psi",
        label="Psi",
        layout=FieldLayout.CELL,
        components=3,  # Multi-component
        dtype=torch.float32,
    )

    cube_field.add_field(phi_field)
    cube_field.add_field(psi_field)

    # Set uniform flux
    phi_f = cube_field.get_field(phi_field)
    phi_f.x[0] = torch.ones(1, 1, N, N, N + 1) * 1.0
    phi_f.y[0] = torch.ones(1, 1, N, N + 1, N) * 1.0
    phi_f.z[0] = torch.ones(1, 1, N + 1, N, N) * 1.0

    # Set uniform multi-component cell field
    psi_c = cube_field.get_field(psi_field)
    psi_c.interior[0] = torch.ones(3, N, N, N) * 2.0

    # Create mock cube node
    mock_node = MagicMock(spec=PyOctreeNode)
    mock_node.field = cube_field
    mock_node.cubecode.neighbor_codes = MagicMock(return_value=[None] * 27)

    # Create context
    ctx = CtxForCubeOperation(
        depth=0,
        bounds=np.array([1, 1, 1], dtype=np.uint64),
        dt=0.1,
        dx=torch.tensor([1.0, 1.0, 1.0]),
        vertices=torch.zeros((8, 3)),
        bcs=[],
    )

    # Build operator
    operator = FVMDivLimitedLinearV(phi_field, psi_field)
    fvmatrix = operator.build(mock_node, ctx)

    # Verify matrix structure
    assert fvmatrix.C == 3  # Multi-component
    assert fvmatrix.N == N
    assert fvmatrix.H == H

    # For uniform fields, source should be zero
    torch.testing.assert_close(
        fvmatrix.source.interior[0],
        torch.zeros(3, N, N, N),
    )


def test_fvm_div_limited_linear_v_build_steepest_gradient_fields():
    """Test FVMDivLimitedLinearV build with steepest gradient fields."""
    # V scheme: compute limiter based on steepest gradient direction
    # and apply the same limiter to all components
    # Setup
    N = 8
    H = 2
    cube_config = CubeConfig(
        interior_width=N, halo_width=H, device=torch.device("cpu")
    )
    cube_field = CubeField(cube_config)

    phi_field = FieldMeta(
        name="phi",
        label="Phi",
        layout=FieldLayout.FACE,
        components=1,
        dtype=torch.float32,
    )
    psi_field = FieldMeta(
        name="psi",
        label="Psi",
        layout=FieldLayout.CELL,
        components=3,  # Multi-component vector field
        dtype=torch.float32,
    )

    cube_field.add_field(phi_field)
    cube_field.add_field(psi_field)

    # Set uniform positive flux
    phi_f = cube_field.get_field(phi_field)
    phi_f.x[0] = torch.ones(1, 1, N, N, N + 1) * 1.0
    phi_f.y[0] = torch.ones(1, 1, N, N + 1, N) * 1.0
    phi_f.z[0] = torch.ones(1, 1, N + 1, N, N) * 1.0

    # Set multi-component field with different gradients in different directions
    # x-direction: large gradient (magnitude ~1.0 per step)
    # y-direction: small gradient (magnitude ~0.1 per step)
    # z-direction: medium gradient (magnitude ~0.5 per step)
    # The steepest gradient is in x-direction,
    # so limiter should be based on x-direction
    psi_c = cube_field.get_field(psi_field)
    # Initialize all cells to avoid boundary issues
    psi_c.interior[0, :, :, :, :] = 0.0

    # Set gradient in x-direction (steepest): component 0 has large gradient
    for i in range(N):
        psi_c.interior[0, 0, :, :, i] = float(
            i
        )  # Large gradient in x, component 0

    # Set gradient in y-direction (small): component 1 has small gradient
    for j in range(N):
        psi_c.interior[0, 1, :, j, :] = (
            float(j) * 0.1
        )  # Small gradient in y, component 1

    # Set gradient in z-direction (medium): component 2 has medium gradient
    for k in range(N):
        psi_c.interior[0, 2, k, :, :] = (
            float(k) * 0.5
        )  # Medium gradient in z, component 2

    # Create mock cube node
    mock_node = MagicMock(spec=PyOctreeNode)
    mock_node.field = cube_field
    mock_node.cubecode.neighbor_codes = MagicMock(return_value=[None] * 27)

    # Create context
    ctx = CtxForCubeOperation(
        depth=0,
        bounds=np.array([1, 1, 1], dtype=np.uint64),
        dt=0.1,
        dx=torch.tensor([1.0, 1.0, 1.0]),
        vertices=torch.zeros((8, 3)),
        bcs=[],
    )

    # Build operator
    operator = FVMDivLimitedLinearV(phi_field, psi_field)
    fvmatrix = operator.build(mock_node, ctx)

    # Verify matrix structure
    assert fvmatrix.C == 3  # Multi-component
    assert fvmatrix.N == N
    assert fvmatrix.H == H

    source = fvmatrix.source.interior[0]  # (C, N, N, N)

    assert torch.all(torch.isfinite(source))
    assert not torch.allclose(source, torch.zeros_like(source), atol=1e-6)

    # The correction should be finite and reasonable
    # For uniform positive flux and gradients,
    # source should be negative (upwind correction)
    # Check that correction is applied (not all zeros)
    max_source = torch.max(torch.abs(source))
    assert max_source > 0.0
    assert max_source < 10.0  # Reasonable upper bound
