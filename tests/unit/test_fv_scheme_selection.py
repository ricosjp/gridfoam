"""What ``fv/schemes/selection.py`` guarantees.

Lookup order
    Field-specific key, then ``default``, then the built-in fallback.
    Tested once on Laplacian; every ``search_*`` is checked against its
    ``default`` and field-specific key.

Div warning
    An omitted ``divSchemes`` dictionary is quiet. A present dictionary
    without a matching key still warns and uses upwind.

Laplacian aliases
    YAML spellings such as ``linear`` become canonical policies.

Dispatch coverage
    Every enum value has a table entry. Aliases are not implementation keys.
"""

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
from gridfoam.meta.config import SimulatorConfig, fvSchemesConfig
from gridfoam.meta.enums import (
    DdtScheme,
    DivScheme,
    FieldRole,
    GradScheme,
    LaplacianScheme,
    SnGradScheme,
)


def _config_with(
    grid: AxisProjectedGrid, family: str, entries: dict[str, str] | None
) -> SimulatorConfig:
    schemes = fvSchemesConfig.model_validate({family: entries})
    return grid.sim_config.model_copy(update={"fvSchemes": schemes})


@pytest.mark.parametrize(
    "entries,expected",
    [
        pytest.param({}, LaplacianScheme.CORRECTED, id="builtin"),
        pytest.param(
            {"default": LaplacianScheme.UNCORRECTED.value},
            LaplacianScheme.UNCORRECTED,
            id="default",
        ),
        pytest.param(
            {
                "default": LaplacianScheme.CORRECTED.value,
                "laplacian(q)": LaplacianScheme.UNCORRECTED.value,
            },
            LaplacianScheme.UNCORRECTED,
            id="field-key",
        ),
    ],
)
def test_lookup_order_is_field_key_then_default_then_builtin(
    small_axis_projected_grid: AxisProjectedGrid,
    entries: dict[str, str] | None,
    expected: LaplacianScheme,
) -> None:
    """Shared ``_lookup`` order. Other operators use the same helper."""
    grid = small_axis_projected_grid
    field = CellField(grid, "q", FieldRole.LOCAL, ())
    config = _config_with(grid, "laplacianSchemes", entries)
    assert search_laplacian_scheme(config, field) == expected


@pytest.mark.parametrize(
    "family,key,fallback,alternative,resolve",
    [
        (
            "divSchemes",
            "div(phi, q)",
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
    "use_field_key", [False, True], ids=["default", "field-key"]
)
def test_each_search_function_reads_its_scheme_family_and_field_key(
    small_axis_projected_grid: AxisProjectedGrid,
    family: str,
    key: str,
    fallback: StrEnum,
    alternative: StrEnum,
    resolve: Callable[..., StrEnum],
    use_field_key: bool,
) -> None:
    """Each operator reads its family and prefers its own key to default."""
    grid = small_axis_projected_grid
    field = CellField(grid, "q", FieldRole.LOCAL, ())
    phi = FaceField(grid, "phi", FieldRole.LOCAL, ())
    entries = (
        {"default": fallback.value, key: alternative.value}
        if use_field_key
        else {"default": alternative.value}
    )
    config = _config_with(grid, family, entries)
    args = (config, phi, field) if family == "divSchemes" else (config, field)
    assert resolve(*args) == alternative


def test_div_warns_when_a_dictionary_exists_without_a_matching_key(
    small_axis_projected_grid: AxisProjectedGrid,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Omitted ``divSchemes`` is quiet; a present miss still uses upwind."""
    grid = small_axis_projected_grid
    field = CellField(grid, "q", FieldRole.LOCAL, ())
    phi = FaceField(grid, "phi", FieldRole.LOCAL, ())
    for entries in ({}, {"div(other, q)": "linear"}):
        caplog.clear()
        config = _config_with(grid, "divSchemes", entries)
        assert search_div_scheme(config, phi, field) == DivScheme.UPWIND
        assert caplog.records
    caplog.clear()
    omitted = _config_with(grid, "divSchemes", None)
    assert search_div_scheme(omitted, phi, field) == DivScheme.UPWIND
    assert not caplog.records


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
def test_laplacian_yaml_spellings_become_canonical_policies(
    small_axis_projected_grid: AxisProjectedGrid,
    value: str,
    canonical: LaplacianScheme,
    harmonic: bool,
    corrected: bool,
) -> None:
    """Alias normalization lives in ``search_laplacian_scheme`` only."""
    grid = small_axis_projected_grid
    field = CellField(grid, "q", FieldRole.LOCAL, ())
    config = _config_with(grid, "laplacianSchemes", {"default": value})
    selected = search_laplacian_scheme(config, field)
    assert selected == canonical
    policy = get_laplacian_scheme(selected)
    assert policy.harmonic is harmonic
    assert policy.corrected is corrected


def test_every_enum_value_has_a_dispatch_table_entry() -> None:
    """A new YAML enum must not fail later at ``get_*_scheme``."""
    assert set(DIV_SCHEMES) == set(DivScheme)
    assert set(GRAD_SCHEMES) == set(GradScheme)
    assert set(SN_GRAD_SCHEMES) == set(SnGradScheme)
    assert set(DDT_SCHEMES) == set(DdtScheme)
    assert set(LAPLACIAN_SCHEMES).isdisjoint(LAPLACIAN_ALIASES)
    assert set(LAPLACIAN_SCHEMES) | set(LAPLACIAN_ALIASES) == set(
        LaplacianScheme
    )
    assert set(LAPLACIAN_ALIASES.values()) <= set(LAPLACIAN_SCHEMES)
