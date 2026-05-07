"""Unit tests for ``gridfoam.meta.enums``."""

from __future__ import annotations

import pytest
import torch
from fluxel import Direction

from gridfoam.meta.enums import (
    DeviceType,
    DomainBoundaryPatch,
    NormType,
    PrecisionType,
)


def test_device_type_to_torch_device_cpu():
    assert DeviceType.CPU.to_torch_device() == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_type_to_torch_device_cuda():
    assert DeviceType.CUDA.to_torch_device() == torch.device("cuda")


def test_norm_type_to_norm_order():
    assert NormType.L_2.to_norm_order() == 2
    assert NormType.L_inf.to_norm_order() == float("inf")


def test_precision_type_to_torch_dtype():
    assert PrecisionType.FLOAT32.to_torch_dtype() is torch.float32
    assert PrecisionType.FLOAT64.to_torch_dtype() is torch.float64


@pytest.mark.parametrize(
    "patch, direction",
    [
        (DomainBoundaryPatch.X_MINUS, Direction.XMinus),
        (DomainBoundaryPatch.X_PLUS, Direction.XPlus),
        (DomainBoundaryPatch.Y_MINUS, Direction.YMinus),
        (DomainBoundaryPatch.Y_PLUS, Direction.YPlus),
        (DomainBoundaryPatch.Z_MINUS, Direction.ZMinus),
        (DomainBoundaryPatch.Z_PLUS, Direction.ZPlus),
    ],
)
def test_domain_boundary_patch_to_direction(
    patch: DomainBoundaryPatch, direction: Direction
):
    assert patch.to_direction() == direction
