"""Tests for FVM div linear operator."""

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
from gridfoam.DNA.scheme.fvm.div._linear import FVMDivLinear


def test_fvm_div_linear_init():
    """Test FVMDivLinear initialization."""
    field = FieldMeta(
        name="test",
        label="Test",
        layout=FieldLayout.CELL,
    )
    operator = FVMDivLinear(field)
    assert operator._psi_fm == field


def test_fvm_div_linear_init_face_error():
    """Test FVMDivLinear raises error for face-centered field."""
    field = FieldMeta(
        name="test",
        label="Test",
        layout=FieldLayout.FACE,
    )
    with pytest.raises(AssertionError):
        FVMDivLinear(field)


def test_fvm_div_linear_build_uniform_field():
    """Test FVMDivLinear build with uniform field."""
    # Setup
    N = 8
    H = 2
    cube_config = CubeConfig(interior_width=N, halo_width=H, device=torch.device("cpu"))
    cube_field = CubeField(cube_config)

    psi_field = FieldMeta(
        name="psi",
        label="Psi",
        layout=FieldLayout.CELL,
        components=3,
        dtype=torch.float32,
    )

    cube_field.add_field(psi_field)

    # Set uniform cell field
    psi_c = cube_field.get_field(psi_field)
    psi_c.raw[0] = torch.ones(3, N+2*H, N+2*H, N+2*H) * 2.0

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
    operator = FVMDivLinear(psi_field)
    fvmatrix = operator.build(mock_node, ctx)

    # Verify matrix structure
    assert fvmatrix.C == 1
    assert fvmatrix.N == N
    assert fvmatrix.H == H

    # For uniform fields, source should be zero
    torch.testing.assert_close(
        fvmatrix.source.interior[0, 0, :, :, :],
        torch.zeros(N, N, N),
        rtol=1e-6,
        atol=1e-6,
    )


def test_fvm_div_linear_build_linear_field():
    """Test FVMDivLinear build with linear field."""
    # Setup
    N = 8
    H = 2
    cube_config = CubeConfig(interior_width=N, halo_width=H, device=torch.device("cpu"))
    cube_field = CubeField(cube_config)

    psi_field = FieldMeta(
        name="psi",
        label="Psi",
        layout=FieldLayout.CELL,
        components=3,
        dtype=torch.float32,
    )

    cube_field.add_field(psi_field)

    # Set linear field in x direction
    psi_c = cube_field.get_field(psi_field)
    for i in range(N+2*H):
        psi_c.raw[0, 0, :, :, i] = float(i)

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
    operator = FVMDivLinear(psi_field)
    fvmatrix = operator.build(mock_node, ctx)

    # For linear fields, source should be -1.0
    torch.testing.assert_close(
        fvmatrix.source.interior[0, 0, :, :, :],
        torch.ones(N, N, N) * -1.0,
        rtol=1e-3,
        atol=1e-3,
    )
