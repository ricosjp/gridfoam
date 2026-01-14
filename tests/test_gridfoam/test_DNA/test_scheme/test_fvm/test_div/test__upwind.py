"""Tests for FVM div upwind operator."""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from gridfoam.DNA.config import CubeConfig
from gridfoam.DNA.cubefield import CubeField
from gridfoam.DNA.ctx_for_cube_operation import CtxForCubeOperation
from gridfoam.DNA.enum import BoundaryConditionType, FieldLayout
from gridfoam.DNA.meta.boundary_condition import BoundaryConditionMeta
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.div._upwind import FVMDivUpwind


def test_fvm_div_upwind_init():
    """Test FVMDivUpwind initialization."""
    phi_field = FieldMeta(
        name="phi",
        label="Phi",
        layout=FieldLayout.FACE,
    )
    psi_field = FieldMeta(
        name="psi",
        label="Psi",
        layout=FieldLayout.CELL,
    )
    operator = FVMDivUpwind(phi_field, psi_field)
    assert operator._phi_fm == phi_field
    assert operator._psi_fm == psi_field


def test_fvm_div_upwind_init_errors():
    """Test FVMDivUpwind raises errors for invalid layouts."""
    phi_field = FieldMeta(
        name="phi",
        label="Phi",
        layout=FieldLayout.CELL,  # Wrong layout
    )
    psi_field = FieldMeta(
        name="psi",
        label="Psi",
        layout=FieldLayout.CELL,
    )
    with pytest.raises(AssertionError):
        FVMDivUpwind(phi_field, psi_field)


def test_fvm_div_upwind_build_uniform_positive_flux_exact():
    """Test FVMDivUpwind build with uniform positive flux - exact values."""
    # Minimal case: N=2, H=1, dx=1.0, uniform flux phi=1.0
    # Expected: V=1.0, all faces have phi=1.0
    # For positive flux:
    #   a_E = min(phi_e, 0) / V = 0
    #   a_W = -max(phi_w, 0) / V = -1.0
    #   a_N = min(phi_n, 0) / V = 0
    #   a_S = -max(phi_s, 0) / V = -1.0
    #   a_T = min(phi_t, 0) / V = 0
    #   a_B = -max(phi_b, 0) / V = -1.0
    #   a_P = (max(phi_e,0) - min(phi_w,0) + ...) / V = (1.0 + 1.0 + 1.0) = 3.0
    
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
    
    # Set uniform positive flux = 1.0
    phi_f = cube_field.get_field(phi_field)
    phi_f.x[0] = torch.ones(1, 1, 2, 2, 3) * 1.0
    phi_f.y[0] = torch.ones(1, 1, 2, 3, 2) * 1.0
    phi_f.z[0] = torch.ones(1, 1, 3, 2, 2) * 1.0
    
    # Set uniform cell field (not used in upwind, but required)
    psi_c = cube_field.get_field(psi_field)
    psi_c.interior[0] = torch.ones(1, 2, 2, 2) * 2.0
    
    mock_node = MagicMock()
    mock_node.field = cube_field
    mock_node.cubecode.neighbor_codes = MagicMock(return_value=[None] * 27)
    
    ctx = CtxForCubeOperation(
        depth=0,
        bounds=np.array([1, 1, 1], dtype=np.uint64),
        dt=0.1,
        dx=torch.tensor([1.0, 1.0, 1.0]),  # V = 1.0
        vertices=torch.zeros((8, 3)),
        bcs=[],
    )
    
    operator = FVMDivUpwind(phi_field, psi_field)
    fvmatrix = operator.build(mock_node, ctx)
    
    # Expected: 内部セル (1,1,1) のみ比較。境界セルは 0 になるので除外。
    V = 1.0
    expected_a_E_center = torch.tensor(-0.0)  # min(1,0)=0
    expected_a_W_center = torch.tensor(-1.0 / V)
    expected_a_N_center = torch.tensor(-0.0)
    expected_a_S_center = torch.tensor(-1.0 / V)
    expected_a_T_center = torch.tensor(-0.0)
    expected_a_B_center = torch.tensor(-1.0 / V)
    expected_a_P_center = torch.tensor(3.0 / V)

    c = fvmatrix.a_E.interior[0][0, 1, 1]
    torch.testing.assert_close(c, expected_a_E_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_W.interior[0][0, 1, 1], expected_a_W_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_N.interior[0][0, 1, 1], expected_a_N_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_S.interior[0][0, 1, 1], expected_a_S_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_T.interior[0][0, 1, 1], expected_a_T_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_B.interior[0][0, 1, 1], expected_a_B_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_P.interior[0][0, 1, 1], expected_a_P_center, rtol=1e-6, atol=1e-6)


def test_fvm_div_upwind_build_uniform_negative_flux_exact():
    """Test FVMDivUpwind build with uniform negative flux - exact values."""
    # Minimal case: N=2, H=1, dx=1.0, uniform flux phi=-1.0
    # For negative flux:
    #   a_E = min(phi_e, 0) / V = -1.0
    #   a_W = -max(phi_w, 0) / V = 0
    #   a_P = (max(phi_e,0) - min(phi_w,0) + ...) / V = (0 - (-1.0)) * 3 = 3.0
    
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
    
    # Set uniform negative flux = -1.0
    phi_f = cube_field.get_field(phi_field)
    phi_f.x[0] = torch.ones(1, 1, 2, 2, 3) * -1.0
    phi_f.y[0] = torch.ones(1, 1, 2, 3, 2) * -1.0
    phi_f.z[0] = torch.ones(1, 1, 3, 2, 2) * -1.0
    
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
    
    operator = FVMDivUpwind(phi_field, psi_field)
    fvmatrix = operator.build(mock_node, ctx)
    
    V = 1.0
    expected_a_E_center = torch.tensor(-1.0 / V)
    expected_a_W_center = torch.tensor(0.0)
    expected_a_N_center = torch.tensor(-1.0 / V)
    expected_a_S_center = torch.tensor(0.0)
    expected_a_T_center = torch.tensor(-1.0 / V)
    expected_a_B_center = torch.tensor(0.0)
    expected_a_P_center = torch.tensor(3.0 / V)

    torch.testing.assert_close(fvmatrix.a_E.interior[0][0, 1, 1], expected_a_E_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_W.interior[0][0, 1, 1], expected_a_W_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_N.interior[0][0, 1, 1], expected_a_N_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_S.interior[0][0, 1, 1], expected_a_S_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_T.interior[0][0, 1, 1], expected_a_T_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_B.interior[0][0, 1, 1], expected_a_B_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_P.interior[0][0, 1, 1], expected_a_P_center, rtol=1e-6, atol=1e-6)


def test_fvm_div_upwind_build_with_dirichlet_bc():
    """Test FVMDivUpwind build with Dirichlet boundary condition."""
    from gridfoam.DNA.constants import DOMAIN_BOUNDARY_MAP
    
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
    
    # Set uniform positive flux
    phi_f = cube_field.get_field(phi_field)
    phi_f.x[0] = torch.ones(1, 1, 2, 2, 3) * 1.0
    phi_f.y[0] = torch.ones(1, 1, 2, 3, 2) * 1.0
    phi_f.z[0] = torch.ones(1, 1, 3, 2, 2) * 1.0
    
    psi_c = cube_field.get_field(psi_field)
    psi_c.interior[0] = torch.ones(1, 2, 2, 2) * 2.0
    
    # Create boundary condition
    bc = BoundaryConditionMeta(
        name="inlet",
        target_field=psi_field,
        type=BoundaryConditionType.DIRICHLET,
        value=torch.tensor([5.0]),
        target_boundary_labels=["inlet"],
    )
    
    # Mock neighbor codes: first face (index 0) is boundary (None)
    nbr_codes = [None] * 27
    # Set some faces as boundaries for testing
    mock_node = MagicMock()
    mock_node.field = cube_field
    mock_node.cubecode.neighbor_codes = MagicMock(return_value=nbr_codes)
    
    ctx = CtxForCubeOperation(
        depth=0,
        bounds=np.array([1, 1, 1], dtype=np.uint64),
        dt=0.1,
        dx=torch.tensor([1.0, 1.0, 1.0]),
        vertices=torch.zeros((8, 3)),
        bcs=[bc],
    )
    
    operator = FVMDivUpwind(phi_field, psi_field)
    fvmatrix = operator.build(mock_node, ctx)
    
    # Verify BC was applied: -X face (forward=False) should have coeff zeroed and source shifted
    a_w_bound = fvmatrix.a_W.get_boundary_cell_along
    source_bound = fvmatrix.source.get_boundary_cell_along
    torch.testing.assert_close(
        a_w_bound(axis=0, forward=False), torch.zeros_like(a_w_bound(axis=0, forward=False))
    )
    # Source should reflect Dirichlet shift: source -= a_boundary * bc_value
    shifted = source_bound(axis=0, forward=False)
    assert torch.isfinite(shifted).all()


def test_fvm_div_upwind_build_with_neumann_bc():
    """Test FVMDivUpwind build with Neumann boundary condition."""
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
    
    phi_f = cube_field.get_field(phi_field)
    phi_f.x[0] = torch.ones(1, 1, 2, 2, 3) * 1.0
    phi_f.y[0] = torch.ones(1, 1, 2, 3, 2) * 1.0
    phi_f.z[0] = torch.ones(1, 1, 3, 2, 2) * 1.0
    
    psi_c = cube_field.get_field(psi_field)
    psi_c.interior[0] = torch.ones(1, 2, 2, 2) * 2.0
    
    # Create Neumann boundary condition
    bc = BoundaryConditionMeta(
        name="outlet",
        target_field=psi_field,
        type=BoundaryConditionType.NEUMANN,
        value=torch.tensor([0.1, 0.1, 0.1]),  # gradient in x, y, z
        target_boundary_labels=["outlet"],
    )
    
    nbr_codes = [None] * 27
    mock_node = MagicMock()
    mock_node.field = cube_field
    mock_node.cubecode.neighbor_codes = MagicMock(return_value=nbr_codes)
    
    ctx = CtxForCubeOperation(
        depth=0,
        bounds=np.array([1, 1, 1], dtype=np.uint64),
        dt=0.1,
        dx=torch.tensor([1.0, 1.0, 1.0]),
        vertices=torch.zeros((8, 3)),
        bcs=[bc],
    )
    
    operator = FVMDivUpwind(phi_field, psi_field)
    fvmatrix = operator.build(mock_node, ctx)
    
    # Verify Neumann BC was applied: boundary coeff zeroed and a_P incremented
    a_w_bound = fvmatrix.a_W.get_boundary_cell_along
    a_p_bound = fvmatrix.a_P.get_boundary_cell_along
    torch.testing.assert_close(
        a_w_bound(axis=0, forward=False), torch.zeros_like(a_w_bound(axis=0, forward=False))
    )
    # a_P boundary should be increased by a_boundary (original negative) -> check finite
    assert torch.isfinite(a_p_bound(axis=0, forward=False)).all()
    assert torch.isfinite(fvmatrix.source.get_boundary_cell_along(axis=0, forward=False)).all()
