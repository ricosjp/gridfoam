"""Configuration precedence and dispatch coverage for FV schemes."""

from collections.abc import Callable
from enum import StrEnum

import pytest

from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.fv.schemes.ddt import DDT_SCHEMES
from gridfoam.fv.schemes.div import DIV_SCHEMES
from gridfoam.fv.schemes.grad import GRAD_SCHEMES
from gridfoam.fv.schemes.laplacian import (
    LAPLACIAN_SCHEMES,
    get_laplacian_scheme,
)
from gridfoam.fv.schemes.selection import (
    LAPLACIAN_ALIASES,
    search_ddt_scheme,
    search_div_scheme,
    search_grad_scheme,
    search_laplacian_scheme,
    search_sn_grad_scheme,
)
from gridfoam.fv.schemes.sn_grad import SN_GRAD_SCHEMES
from gridfoam.meta.config import fvSchemesConfig
from gridfoam.meta.enums import (
    DdtScheme,
    DivScheme,
    FieldRole,
    GradScheme,
    LaplacianScheme,
    SnGradScheme,
)


@pytest.mark.parametrize(
    "config_name,key,builtin,alternative,resolve",
    [
        (
            "divSchemes",
            "div(phi,q)",
            DivScheme.UPWIND,
            DivScheme.LINEAR,
            search_div_scheme,
        ),
        (
            "gradSchemes",
            "grad(q)",
            GradScheme.LEASTSQUARE,
            GradScheme.LINEAR,
            search_grad_scheme,
        ),
        (
            "laplacianSchemes",
            "laplacian(q)",
            LaplacianScheme.CORRECTED,
            LaplacianScheme.UNCORRECTED,
            search_laplacian_scheme,
        ),
        (
            "snGradSchemes",
            "snGrad(q)",
            SnGradScheme.CORRECTED,
            SnGradScheme.UNCORRECTED,
            search_sn_grad_scheme,
        ),
        (
            "ddtSchemes",
            "ddt(q)",
            DdtScheme.EULER,
            DdtScheme.BACKWARD,
            search_ddt_scheme,
        ),
    ],
)
@pytest.mark.parametrize(
    "selection", ["absent", "empty", "unrelated", "default", "specific"]
)
def test_lookup_precedence(
    small_axis_projected_grid: AxisProjectedGrid,
    config_name: str,
    key: str,
    builtin: StrEnum,
    alternative: StrEnum,
    resolve: Callable[..., StrEnum],
    selection: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    grid = small_axis_projected_grid
    field = CellField(grid, "q", FieldRole.LOCAL, ())
    phi = FaceField(grid, "phi", FieldRole.LOCAL, ())
    entries = {
        "absent": None,
        "empty": {},
        "unrelated": {"another(field)": alternative.value},
        "default": {"default": alternative.value},
        "specific": {"default": builtin.value, key: alternative.value},
    }[selection]
    schemes = fvSchemesConfig.model_validate({config_name: entries})
    config = grid.sim_config.model_copy(update={"fvSchemes": schemes})
    args = (
        (config, phi, field) if config_name == "divSchemes" else (config, field)
    )
    expected = alternative if selection in ("default", "specific") else builtin
    assert resolve(*args) == expected
    # An omitted div dictionary is quiet; an incomplete one still warns.
    assert bool(caplog.records) == (
        config_name == "divSchemes" and selection in ("empty", "unrelated")
    )


@pytest.mark.parametrize("key", ["default", "laplacian(q)"])
@pytest.mark.parametrize(
    "value,canonical,harmonic,corrected",
    [
        ("linear", LaplacianScheme.CORRECTED, False, True),
        ("corrected", LaplacianScheme.CORRECTED, False, True),
        ("uncorrected", LaplacianScheme.UNCORRECTED, False, False),
        ("Gauss linear corrected", LaplacianScheme.CORRECTED, False, True),
        ("Gauss linear uncorrected", LaplacianScheme.UNCORRECTED, False, False),
        (
            "Gauss harmonic corrected",
            LaplacianScheme.GAUSS_HARMONIC_CORRECTED,
            True,
            True,
        ),
        (
            "Gauss harmonic uncorrected",
            LaplacianScheme.GAUSS_HARMONIC_UNCORRECTED,
            True,
            False,
        ),
    ],
)
def test_laplacian_aliases_and_policies(
    small_axis_projected_grid: AxisProjectedGrid,
    key: str,
    value: str,
    canonical: LaplacianScheme,
    harmonic: bool,
    corrected: bool,
) -> None:
    grid = small_axis_projected_grid
    field = CellField(grid, "q", FieldRole.LOCAL, ())
    schemes = fvSchemesConfig.model_validate({"laplacianSchemes": {key: value}})
    config = grid.sim_config.model_copy(update={"fvSchemes": schemes})
    selected = search_laplacian_scheme(config, field)
    assert selected == canonical
    policy = get_laplacian_scheme(selected)
    assert policy.harmonic is harmonic
    assert policy.corrected is corrected


def test_every_configured_scheme_has_an_implementation() -> None:
    # A newly accepted configuration value must not fail later at dispatch.
    assert set(DIV_SCHEMES) == set(DivScheme)
    assert set(GRAD_SCHEMES) == set(GradScheme)
    assert set(SN_GRAD_SCHEMES) == set(SnGradScheme)
    assert set(DDT_SCHEMES) == set(DdtScheme)
    assert set(LAPLACIAN_SCHEMES).isdisjoint(LAPLACIAN_ALIASES)
    assert set(LAPLACIAN_SCHEMES) | set(LAPLACIAN_ALIASES) == set(
        LaplacianScheme
    )
    assert set(LAPLACIAN_ALIASES.values()) <= set(LAPLACIAN_SCHEMES)
