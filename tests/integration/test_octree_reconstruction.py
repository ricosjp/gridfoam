import importlib

import torch
from pytest import MonkeyPatch
from tests.conftest import small_gridfoam_config

from gridfoam.core.field import CellField
from gridfoam.core.grid.base import IGridBase
from gridfoam.core.grid.factory import create_grid
from gridfoam.fv.fvc.grad import grad
from gridfoam.fv.fvc.interpolate import interpolate
from gridfoam.fv.fvc.reconstruction import (
    linear_internal_face_values,
    single_internal_mask,
)
from gridfoam.fv.fvc.sn_grad import sn_grad
from gridfoam.meta.config import GridfoamConfig, RefinementRegionConfig
from gridfoam.meta.enums import FieldRole, GradScheme

grad_module = importlib.import_module("gridfoam.fv.fvc.grad")


def _refined_config(*, grad_scheme: GradScheme | None = None) -> GridfoamConfig:
    config = small_gridfoam_config()
    config = config.model_copy(
        update={
            "fluxel": config.fluxel.model_copy(
                update={
                    "root_resolution": [4, 4, 1],
                    "refinement_regions": [
                        RefinementRegionConfig(
                            name="center",
                            min=[0.25, 0.25, 0.0],
                            max=[0.75, 0.75, 0.1],
                            level=1,
                        )
                    ],
                }
            )
        }
    )
    if grad_scheme is not None:
        config = config.model_copy(
            update={
                "simulator": config.simulator.model_copy(
                    update={
                        "fvSchemes": config.simulator.fvSchemes.model_copy(
                            update={"gradSchemes": {"default": grad_scheme}}
                        )
                    }
                )
            }
        )
    return config


def _refined_grid(*, grad_scheme: GradScheme | None = None) -> IGridBase:
    return create_grid(_refined_config(grad_scheme=grad_scheme))


def _linear_scalar_field(
    grid: IGridBase,
) -> tuple[CellField, torch.Tensor]:
    field = CellField(
        grid,
        name="linear_scalar",
        role=FieldRole.LOCAL,
        num_components=1,
        export=False,
    )
    gradient = torch.tensor(
        [2.0, -3.0, 5.0], dtype=grid.dtype, device=grid.device
    )
    field.data = (grid.cell_centers @ gradient + 7.0).reshape(-1, 1)
    return field, gradient


def test_linear_internal_face_values_is_exact_on_uniform_mesh():
    grid = create_grid(small_gridfoam_config())
    field, gradient = _linear_scalar_field(grid)
    single_mask = single_internal_mask(grid)

    linear = linear_internal_face_values(field)
    expected = (grid.face_centers[single_mask] @ gradient + 7.0).reshape(-1, 1)

    torch.testing.assert_close(linear, expected, atol=1e-12, rtol=1e-12)


def test_interpolate_uses_linear_internal_face_values():
    grid = create_grid(small_gridfoam_config())
    field, _ = _linear_scalar_field(grid)

    interpolated = interpolate(field)
    linear = linear_internal_face_values(field)
    torch.testing.assert_close(
        interpolated.single_data, linear, atol=1e-12, rtol=1e-12
    )


def test_sn_grad_default_linear_scheme_does_not_use_leastsquare(
    monkeypatch: MonkeyPatch,
):
    def fail_leastsquare(_field: CellField) -> torch.Tensor:
        raise AssertionError("leastSquare reconstruction should not be used")

    monkeypatch.setattr(grad_module, "least_square_grad_data", fail_leastsquare)

    grid = _refined_grid()
    field, _ = _linear_scalar_field(grid)

    sn_grad(field)


def test_sn_grad_is_exact_with_leastsquare_grad_scheme_on_octree_interfaces():
    grid = _refined_grid(grad_scheme=GradScheme.LEASTSQUARE)
    field, gradient = _linear_scalar_field(grid)

    sn_grad_result = sn_grad(field)
    expected = gradient[grid.axis[sn_grad_result.single_mask]].reshape(-1, 1)

    torch.testing.assert_close(
        sn_grad_result.single_data, expected, atol=1e-12, rtol=1e-12
    )


def test_sn_grad_reuses_cached_grad_field():
    grid = _refined_grid(grad_scheme=GradScheme.LEASTSQUARE)
    field_direct, gradient = _linear_scalar_field(grid)

    sn_grad_direct = sn_grad(field_direct)

    field_cached, _ = _linear_scalar_field(grid)
    grad(field_cached)
    sn_grad_cached = sn_grad(field_cached)

    expected = gradient[grid.axis[sn_grad_direct.single_mask]].reshape(-1, 1)
    torch.testing.assert_close(
        sn_grad_direct.single_data, expected, atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        sn_grad_cached.single_data, expected, atol=1e-12, rtol=1e-12
    )
