"""Tests for FVM div limited linear operator."""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from gridfoam.DNA.config import CubeConfig
from gridfoam.DNA.cubefield import CubeField
from gridfoam.DNA.ctx_for_cube_operation import CtxForCubeOperation
from gridfoam.DNA.enum import FieldLayout
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.div._limited_linear import FVMDivLimitedLinear


def test_fvm_div_limited_linear_init():
    """Test FVMDivLimitedLinear initialization."""
    phi_field = FieldMeta(
        name="phi",
        label="Phi",
        layout=FieldLayout.FACE,
    )
    psi_field = FieldMeta(
        name="psi",
        label="Psi",
        layout=FieldLayout.CELL,
        components=1,
    )
    operator = FVMDivLimitedLinear(phi_field, psi_field)
    assert operator._phi_fm == phi_field
    assert operator._psi_fm == psi_field
    assert operator._limited_linear is not None


def test_fvm_div_limited_linear_init_errors():
    """Test FVMDivLimitedLinear raises errors for invalid layouts."""
    phi_field = FieldMeta(
        name="phi",
        label="Phi",
        layout=FieldLayout.CELL,  # Wrong layout
    )
    psi_field = FieldMeta(
        name="psi",
        label="Psi",
        layout=FieldLayout.CELL,
        components=1,
    )
    with pytest.raises(AssertionError):
        FVMDivLimitedLinear(phi_field, psi_field)

    phi_field = FieldMeta(
        name="phi",
        label="Phi",
        layout=FieldLayout.FACE,
    )
    psi_field = FieldMeta(
        name="psi",
        label="Psi",
        layout=FieldLayout.FACE,  # Wrong layout
        components=1,
    )
    with pytest.raises(AssertionError):
        FVMDivLimitedLinear(phi_field, psi_field)


def test_fvm_div_limited_linear_init_components_error():
    """Test FVMDivLimitedLinear raises error for multi-component field."""
    phi_field = FieldMeta(
        name="phi",
        label="Phi",
        layout=FieldLayout.FACE,
    )
    psi_field = FieldMeta(
        name="psi",
        label="Psi",
        layout=FieldLayout.CELL,
        components=3,  # Should be 1 for LimitedLinear
    )
    with pytest.raises(AssertionError):
        FVMDivLimitedLinear(phi_field, psi_field)


def test_fvm_div_limited_linear_build_uniform_fields_exact():
    """Test FVMDivLimitedLinear build with uniform fields - exact values."""
    # Minimal case: N=2, H=1, uniform fields
    # For uniform fields, TVD correction should be zero
    
    cube_config = CubeConfig(interior_width=2, halo_width=1, device=torch.device("cpu"))
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
        components=1,
        dtype=torch.float32,
    )
    
    cube_field.add_field(phi_field)
    cube_field.add_field(psi_field)
    
    # Set uniform flux
    phi_f = cube_field.get_field(phi_field)
    phi_f.x[0] = torch.ones(1, 1, 2, 2, 3) * 1.0
    phi_f.y[0] = torch.ones(1, 1, 2, 3, 2) * 1.0
    phi_f.z[0] = torch.ones(1, 1, 3, 2, 2) * 1.0
    
    # Set uniform cell field
    psi_c = cube_field.get_field(psi_field)
    psi_c.interior[0] = torch.ones(1, 2, 2, 2) * 2.0
    
    mock_node = MagicMock()
    mock_node.field = cube_field
    mock_node.cubecode.neighbor_codes = MagicMock(return_value=[None] * 27)
    
    ctx = CtxForCubeOperation(
        depth=0,
        bounds=np.array([1, 1, 1], dtype=np.uint64),
        dt=0.1,
        dx=torch.tensor([1.0, 1.0, 1.0]),
        vertices=torch.zeros((8, 3)),
        bcs=[],
    )
    
    operator = FVMDivLimitedLinear(phi_field, psi_field)
    fvmatrix = operator.build(mock_node, ctx)
    
    # For uniform fields, source (TVD correction) should be exactly zero
    torch.testing.assert_close(
        fvmatrix.source.interior[0][0, 1, 1, 1],
        torch.tensor(0.0),
        rtol=1e-6,
        atol=1e-6,
    )


def test_fvm_div_limited_linear_tvd_correction_gradient_sign_change():
    """Test FVMDivLimitedLinear TVD correction with gradient sign change."""
    # Create field with gradient sign change: psi = [0, 1, 2, 1]
    # At cell 1-2 boundary: delta_minus=1, delta_plus=1 (same sign, correction applied)
    # At cell 2-3 boundary: delta_minus=1, delta_plus=-1 (opposite sign, no correction)
    
    cube_config = CubeConfig(interior_width=4, halo_width=1, device=torch.device("cpu"))
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
        components=1,
        dtype=torch.float32,
    )
    
    cube_field.add_field(phi_field)
    cube_field.add_field(psi_field)
    
    # Set uniform positive flux
    phi_f = cube_field.get_field(phi_field)
    phi_f.x[0] = torch.ones(1, 1, 4, 4, 5) * 1.0
    phi_f.y[0] = torch.ones(1, 1, 4, 5, 4) * 1.0
    phi_f.z[0] = torch.ones(1, 1, 5, 4, 4) * 1.0
    
    # Set field with gradient sign change in x direction
    psi_c = cube_field.get_field(psi_field)
    psi_c.interior[0, 0, :, :, 0] = 0.0
    psi_c.interior[0, 0, :, :, 1] = 1.0
    psi_c.interior[0, 0, :, :, 2] = 2.0
    psi_c.interior[0, 0, :, :, 3] = 1.0
    
    mock_node = MagicMock()
    mock_node.field = cube_field
    mock_node.cubecode.neighbor_codes = MagicMock(return_value=[None] * 27)
    
    ctx = CtxForCubeOperation(
        depth=0,
        bounds=np.array([1, 1, 1], dtype=np.uint64),
        dt=0.1,
        dx=torch.tensor([1.0, 1.0, 1.0]),
        vertices=torch.zeros((8, 3)),
        bcs=[],
    )
    
    operator = FVMDivLimitedLinear(phi_field, psi_field)
    fvmatrix = operator.build(mock_node, ctx)
    
    # 中心セル (1,1,1) では滑らかな勾配 → 補正あり
    center_source = fvmatrix.source.interior[0][0, 1, 1, 1]
    assert torch.isfinite(center_source)
    # 符号反転を跨ぐセル(3方向の境界近傍)は補正が小さい/ゼロ
    sign_change_source = fvmatrix.source.interior[0][0, 1, 1, 2]
    assert torch.abs(sign_change_source) <= torch.abs(center_source) + 1e-5


def test_fvm_div_limited_linear_tvd_correction_steep_gradient():
    """Test FVMDivLimitedLinear TVD correction with steep gradient."""
    # Create field with steep gradient: psi = [0, 0.1, 1.0, 1.1]
    # LimitedLinear limiter should clip the correction
    
    cube_config = CubeConfig(interior_width=4, halo_width=1, device=torch.device("cpu"))
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
        components=1,
        dtype=torch.float32,
    )
    
    cube_field.add_field(phi_field)
    cube_field.add_field(psi_field)
    
    phi_f = cube_field.get_field(phi_field)
    phi_f.x[0] = torch.ones(1, 1, 4, 4, 5) * 1.0
    phi_f.y[0] = torch.ones(1, 1, 4, 5, 4) * 1.0
    phi_f.z[0] = torch.ones(1, 1, 5, 4, 4) * 1.0
    
    # Set field with steep gradient
    psi_c = cube_field.get_field(psi_field)
    psi_c.interior[0, 0, :, :, 0] = 0.0
    psi_c.interior[0, 0, :, :, 1] = 0.1
    psi_c.interior[0, 0, :, :, 2] = 1.0
    psi_c.interior[0, 0, :, :, 3] = 1.1
    
    mock_node = MagicMock()
    mock_node.field = cube_field
    mock_node.cubecode.neighbor_codes = MagicMock(return_value=[None] * 27)
    
    ctx = CtxForCubeOperation(
        depth=0,
        bounds=np.array([1, 1, 1], dtype=np.uint64),
        dt=0.1,
        dx=torch.tensor([1.0, 1.0, 1.0]),
        vertices=torch.zeros((8, 3)),
        bcs=[],
    )
    
    operator = FVMDivLimitedLinear(phi_field, psi_field)
    fvmatrix = operator.build(mock_node, ctx)
    
    # Source should be computed and finite
    assert torch.isfinite(fvmatrix.source.interior[0]).all()
    
    # 補正はクリップされ、滑らかケースより小さい値になる
    source = fvmatrix.source.interior[0][0, 1, 1, 1]
    assert torch.isfinite(source)
    assert torch.abs(source) < 1.0  # 限界より十分小さいはず
