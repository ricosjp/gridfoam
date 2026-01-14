"""Tests for FVM div limited linear V operator."""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from gridfoam.DNA.config import CubeConfig
from gridfoam.DNA.cubefield import CubeField
from gridfoam.DNA.ctx_for_cube_operation import CtxForCubeOperation
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
        components=3,  # Multi-component
        dtype=torch.float32,
    )
    
    cube_field.add_field(phi_field)
    cube_field.add_field(psi_field)
    
    # Set uniform flux
    phi_f = cube_field.get_field(phi_field)
    phi_f.x[0] = torch.ones(1, 1, 4, 4, 5) * 1.0
    phi_f.y[0] = torch.ones(1, 1, 4, 5, 4) * 1.0
    phi_f.z[0] = torch.ones(1, 1, 5, 4, 4) * 1.0
    
    # Set uniform multi-component cell field
    psi_c = cube_field.get_field(psi_field)
    psi_c.interior[0] = torch.ones(3, 4, 4, 4) * 2.0
    
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
    operator = FVMDivLimitedLinearV(phi_field, psi_field)
    fvmatrix = operator.build(mock_node, ctx)
    
    # Verify matrix structure
    assert fvmatrix.C == 3  # Multi-component
    assert fvmatrix.N == 4
    assert fvmatrix.H == 1
    
    # For uniform fields, source should be zero
    torch.testing.assert_close(
        fvmatrix.source.interior[0],
        torch.zeros(3, 4, 4, 4),
        rtol=1e-5,
    )
