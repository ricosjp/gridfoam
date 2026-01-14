"""Tests for FVM div linear operator."""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from gridfoam.DNA.config import CubeConfig
from gridfoam.DNA.cubefield import CubeField
from gridfoam.DNA.ctx_for_cube_operation import CtxForCubeOperation
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
    cube_config = CubeConfig(interior_width=4, halo_width=1, device=torch.device("cpu"))
    cube_field = CubeField(cube_config)
    
    psi_field = FieldMeta(
        name="psi",
        label="Psi",
        layout=FieldLayout.CELL,
        components=1,
        dtype=torch.float32,
    )
    
    cube_field.add_field(psi_field)
    
    # Set uniform cell field
    psi_c = cube_field.get_field(psi_field)
    psi_c.interior[0] = torch.ones(1, 4, 4, 4) * 2.0
    
    # Create mock cube node
    mock_node = MagicMock()
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
    assert fvmatrix.N == 4
    assert fvmatrix.H == 1
    
    # 内部セル(1,1,1)のみ比較（境界は face_average の影響でずれる）
    torch.testing.assert_close(
        fvmatrix.source.interior[0][0, 1, 1, 1],
        torch.tensor(0.0),
        rtol=1e-6,
        atol=1e-6,
    )


def test_fvm_div_linear_build_linear_field():
    """Test FVMDivLinear build with linear field."""
    # Setup
    cube_config = CubeConfig(interior_width=4, halo_width=1, device=torch.device("cpu"))
    cube_field = CubeField(cube_config)
    
    psi_field = FieldMeta(
        name="psi",
        label="Psi",
        layout=FieldLayout.CELL,
        components=1,
        dtype=torch.float32,
    )
    
    cube_field.add_field(psi_field)
    
    # Set linear field in x direction
    psi_c = cube_field.get_field(psi_field)
    for i in range(4):
        psi_c.interior[0, 0, :, :, i] = float(i)
    
    # Create mock cube node
    mock_node = MagicMock()
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
    
    # 内部セル(1,1,1)で理論値 -1.0 を比較（source は -div）
    torch.testing.assert_close(
        fvmatrix.source.interior[0][0, 1, 1, 1],
        torch.tensor(-1.0),
        rtol=1e-3,
        atol=1e-3,
    )
