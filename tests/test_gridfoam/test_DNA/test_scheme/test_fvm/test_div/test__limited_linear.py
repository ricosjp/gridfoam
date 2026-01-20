"""Tests for FVM div limited linear operator."""

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
        components=1,
        dtype=torch.float32,
    )

    cube_field.add_field(phi_field)
    cube_field.add_field(psi_field)

    # Set uniform flux
    phi_f = cube_field.get_field(phi_field)
    phi_f.x[0] = torch.ones(1, 1, N, N, N + 1) * 1.0
    phi_f.y[0] = torch.ones(1, 1, N, N + 1, N) * 1.0
    phi_f.z[0] = torch.ones(1, 1, N + 1, N, N) * 1.0

    # Set uniform cell field
    psi_c = cube_field.get_field(psi_field)
    psi_c.interior[0] = torch.ones(1, N, N, N) * 2.0

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

    operator = FVMDivLimitedLinear(phi_field, psi_field)
    fvmatrix = operator.build(mock_node, ctx)

    # For uniform fields, source (TVD correction) should be exactly zero
    torch.testing.assert_close(
        fvmatrix.source.interior[0, 0, :, :, :],
        torch.zeros(N, N, N),
        rtol=1e-6,
        atol=1e-6,
    )


def test_fvm_div_limited_linear_tvd_correction_gradient_sign_change():
    """Test FVMDivLimitedLinear TVD correction with gradient sign change."""
    # Create field with gradient sign change: psi = [0, 1, 2, 1]
    # At cell 1-2 boundary: delta_minus=1, delta_plus=1 (same sign, correction applied)
    # At cell 2-3 boundary: delta_minus=1, delta_plus=-1 (opposite sign, no correction)
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
        components=1,
        dtype=torch.float32,
    )

    cube_field.add_field(phi_field)
    cube_field.add_field(psi_field)

    # Set uniform positive flux
    phi_f = cube_field.get_field(phi_field)
    phi_f.x[0] = torch.ones(1, 1, N, N, N + 1) * 1.0
    phi_f.y[0] = torch.ones(1, 1, N, N + 1, N) * 1.0
    phi_f.z[0] = torch.ones(1, 1, N + 1, N, N) * 1.0

    psi_c = cube_field.get_field(psi_field)
    # Set values in x direction: [0, 1, 2, 1, ...] with extension
    # Initialize all cells first to avoid uninitialized values
    psi_c.interior[0, 0, :, :, :] = 1.0  # Default value
    # Then set the gradient pattern
    psi_c.interior[0, 0, :, :, 0] = 0.0
    psi_c.interior[0, 0, :, :, 1] = 1.0
    psi_c.interior[0, 0, :, :, 2] = 2.0
    psi_c.interior[0, 0, :, :, 3] = 1.0

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

    operator = FVMDivLimitedLinear(phi_field, psi_field)
    fvmatrix = operator.build(mock_node, ctx)
    monotonic_source = fvmatrix.source.interior[0, 0, :, :, [0, 1, 3]]
    assert torch.all(monotonic_source < 0.5)
    sign_change_source = fvmatrix.source.interior[0, 0, :, :, 2]  # (N, N)
    assert torch.all(sign_change_source == 0.5)
