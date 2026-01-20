"""Tests for FVM div upwind operator."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from gridfoam.DNA._grid._grid import PyOctreeNode
from gridfoam.DNA.config import CubeConfig
from gridfoam.DNA.ctx_for_cube_operation import CtxForCubeOperation
from gridfoam.DNA.cubefield import CubeField
from gridfoam.DNA.enum import Axis, BoundaryConditionType, FieldLayout
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

    N = 8
    H = 2
    cube_config = CubeConfig(interior_width=N, halo_width=H, device=torch.device("cpu"))
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
    phi_f.x[0] = torch.ones(1, 1, N, N, N+1) * 1.0
    phi_f.y[0] = torch.ones(1, 1, N, N+1, N) * 1.0
    phi_f.z[0] = torch.ones(1, 1, N+1, N, N) * 1.0

    # Set uniform cell field (not used in upwind, but required)
    psi_c = cube_field.get_field(psi_field)
    psi_c.raw[0] = torch.ones(1, N+2*H, N+2*H, N+2*H) * 2.0

    mock_node = MagicMock(spec=PyOctreeNode)
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

    V = 1.0
    expected_a_E_center = torch.zeros(N, N, N)  # min(1,0)=0
    expected_a_W_center = torch.ones(N, N, N) * -1.0 / V
    expected_a_N_center = torch.zeros(N, N, N)
    expected_a_S_center = torch.ones(N, N, N) * -1.0 / V
    expected_a_T_center = torch.zeros(N, N, N)
    expected_a_B_center = torch.ones(N, N, N) * -1.0 / V
    expected_a_P_center = torch.ones(N, N, N) * 3.0 / V

    torch.testing.assert_close(fvmatrix.a_E.interior[0, 0, :, :, :], expected_a_E_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_W.interior[0, 0, :, :, :], expected_a_W_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_N.interior[0, 0, :, :, :], expected_a_N_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_S.interior[0, 0, :, :, :], expected_a_S_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_T.interior[0, 0, :, :, :], expected_a_T_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_B.interior[0, 0, :, :, :], expected_a_B_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_P.interior[0, 0, :, :, :], expected_a_P_center, rtol=1e-6, atol=1e-6)


def test_fvm_div_upwind_build_uniform_negative_flux_exact():
    """Test FVMDivUpwind build with uniform negative flux - exact values."""
    # Minimal case: N=2, H=1, dx=1.0, uniform flux phi=-1.0
    # For negative flux:
    #   a_E = min(phi_e, 0) / V = -1.0
    #   a_W = -max(phi_w, 0) / V = 0
    #   a_P = (max(phi_e,0) - min(phi_w,0) + ...) / V = (0 - (-1.0)) * 3 = 3.0

    N = 8
    H = 2
    cube_config = CubeConfig(interior_width=N, halo_width=H, device=torch.device("cpu"))
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
    phi_f.x[0] = torch.ones(1, 1, N, N, N+1) * -1.0
    phi_f.y[0] = torch.ones(1, 1, N, N+1, N) * -1.0
    phi_f.z[0] = torch.ones(1, 1, N+1, N, N) * -1.0

    psi_c = cube_field.get_field(psi_field)
    psi_c.raw[0] = torch.ones(1, N+2*H, N+2*H, N+2*H) * 2.0

    mock_node = MagicMock(spec=PyOctreeNode)
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
    expected_a_E_center = torch.ones(N, N, N) * -1.0 / V
    expected_a_W_center = torch.zeros(N, N, N)
    expected_a_N_center = torch.ones(N, N, N) * -1.0 / V
    expected_a_S_center = torch.zeros(N, N, N)
    expected_a_T_center = torch.ones(N, N, N) * -1.0 / V
    expected_a_B_center = torch.zeros(N, N, N)
    expected_a_P_center = torch.ones(N, N, N) * 3.0 / V

    torch.testing.assert_close(fvmatrix.a_E.interior[0, 0, :, :, :], expected_a_E_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_W.interior[0, 0, :, :, :], expected_a_W_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_N.interior[0, 0, :, :, :], expected_a_N_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_S.interior[0, 0, :, :, :], expected_a_S_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_T.interior[0, 0, :, :, :], expected_a_T_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_B.interior[0, 0, :, :, :], expected_a_B_center, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fvmatrix.a_P.interior[0, 0, :, :, :], expected_a_P_center, rtol=1e-6, atol=1e-6)


def test_fvm_div_upwind_build_with_dirichlet_bc():
    """Test FVMDivUpwind build with Dirichlet boundary condition."""

    N = 8
    H = 2
    cube_config = CubeConfig(interior_width=N, halo_width=H, device=torch.device("cpu"))
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
    phi_f.x[0] = torch.ones(1, 1, N, N, N+1) * 1.0
    phi_f.y[0] = torch.ones(1, 1, N, N+1, N) * 1.0
    phi_f.z[0] = torch.ones(1, 1, N+1, N, N) * 1.0

    psi_c = cube_field.get_field(psi_field)
    psi_c.raw[0] = torch.ones(1, N+2*H, N+2*H, N+2*H)

    # Create boundary condition
    bc = BoundaryConditionMeta(
        name="inlet",
        target_field=psi_field,
        type=BoundaryConditionType.DIRICHLET,
        value=torch.tensor([5.0]),
        target_boundary_labels=["domainX-"],
    )

    # Mock neighbor codes: first face (index 0) is boundary (None)
    nbr_codes = [None] * 27
    # Set some faces as boundaries for testing
    mock_node = MagicMock(spec=PyOctreeNode)
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
    a_w_bound = fvmatrix.a_W.get_boundary_cell_along(axis=Axis.X, forward=False)
    source_bound = fvmatrix.source.get_boundary_cell_along(axis=Axis.X, forward=False)
    torch.testing.assert_close(
        a_w_bound, torch.zeros_like(a_w_bound)
    )
    # Source should reflect Dirichlet shift: source -= a_boundary * bc_value
    torch.testing.assert_close(source_bound[0, 0], torch.ones(N, N) * 5.0, rtol=1e-6, atol=1e-6)


def test_fvm_div_upwind_build_with_neumann_bc():
    """Test FVMDivUpwind build with Neumann boundary condition."""
    N = 8
    H = 2
    cube_config = CubeConfig(interior_width=N, halo_width=H, device=torch.device("cpu"))
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
    for i in range(N+1):
        phi_f.x[0, 0, :, :, i] = -float(i)
    phi_f.y[0] = torch.ones(1, 1, N, N+1, N) * 1.0
    phi_f.z[0] = torch.ones(1, 1, N+1, N, N) * 1.0

    psi_c = cube_field.get_field(psi_field)
    psi_c.raw[0] = torch.ones(1, N+2*H, N+2*H, N+2*H)

    # Create Neumann boundary condition
    bc = BoundaryConditionMeta(
        name="outlet",
        target_field=psi_field,
        type=BoundaryConditionType.NEUMANN,
        value=torch.tensor([0.1, 0.1, 0.1]),  # gradient in x, y, z
        target_boundary_labels=["domainX+"],
    )

    nbr_codes = [None] * 27
    mock_node = MagicMock(spec=PyOctreeNode)
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
    a_e_bound = fvmatrix.a_E.get_boundary_cell_along(axis=Axis.X, forward=True)
    a_p_bound = fvmatrix.a_P.get_boundary_cell_along(axis=Axis.X, forward=True)
    source_bound = fvmatrix.source.get_boundary_cell_along(axis=Axis.X, forward=True)
    torch.testing.assert_close(
        a_e_bound, torch.zeros_like(a_e_bound)
    )
    # a_P boundary should be increased by a_boundary (original negative) -> check finite
    torch.testing.assert_close(a_p_bound[0, 0], torch.ones(N, N) * 1.0, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(source_bound[0, 0], torch.ones(N, N) * 8.0*0.5*0.1, rtol=1e-6, atol=1e-6)
