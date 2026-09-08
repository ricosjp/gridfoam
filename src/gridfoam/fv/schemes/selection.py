"""Configuration lookup for all selectable finite-volume schemes.

Each operator uses its specific key, then ``default``, then its built-in
default. Alias normalization belongs here; numerical implementations and
dispatch tables belong in the corresponding ``schemes`` module.
"""

import logging
from collections.abc import Mapping

from gridfoam.core.field import CellField, FaceField
from gridfoam.meta.config import SimulatorConfig
from gridfoam.meta.enums import (
    DdtScheme,
    DivScheme,
    GradScheme,
    LaplacianScheme,
    SnGradScheme,
)

logger = logging.getLogger(__name__)

DEFAULT_DIV_SCHEME = DivScheme.UPWIND
DEFAULT_GRAD_SCHEME = GradScheme.LEASTSQUARE
DEFAULT_LAPLACIAN_SCHEME = LaplacianScheme.CORRECTED
DEFAULT_SN_GRAD_SCHEME = SnGradScheme.CORRECTED
DEFAULT_DDT_SCHEME = DdtScheme.EULER

LAPLACIAN_ALIASES: dict[LaplacianScheme, LaplacianScheme] = {
    LaplacianScheme.LINEAR: LaplacianScheme.CORRECTED,
    LaplacianScheme.GAUSS_LINEAR_CORRECTED: LaplacianScheme.CORRECTED,
    LaplacianScheme.GAUSS_LINEAR_UNCORRECTED: LaplacianScheme.UNCORRECTED,
}


def _lookup[Scheme](
    schemes: Mapping[str, Scheme] | None,
    key: str,
    default: Scheme,
    *,
    missing_warning: str | None = None,
) -> Scheme:
    if schemes is None:
        return default
    scheme = schemes.get(key)
    if scheme is None:
        scheme = schemes.get("default")
    if scheme is None:
        if missing_warning is not None:
            logger.warning(missing_warning, key)
        return default
    return scheme


def search_div_scheme(
    sim_config: SimulatorConfig, phi: FaceField, field: CellField
) -> DivScheme:
    """Resolve ``div(<flux>, <field>)``, preserving the upwind warning."""
    return _lookup(
        sim_config.fvSchemes.divSchemes,
        f"div({phi.name}, {field.name})",
        DEFAULT_DIV_SCHEME,
        missing_warning="Div scheme for %s not found. Using UPWIND scheme.",
    )


def search_grad_scheme(
    sim_config: SimulatorConfig, field: CellField
) -> GradScheme:
    """Resolve ``grad(<field>)``."""
    return _lookup(
        sim_config.fvSchemes.gradSchemes,
        f"grad({field.name})",
        DEFAULT_GRAD_SCHEME,
    )


def search_laplacian_scheme(
    sim_config: SimulatorConfig, field: CellField
) -> LaplacianScheme:
    """Resolve ``laplacian(<field>)`` to a canonical scheme.

    YAML aliases such as ``linear`` are normalized here so dispatch tables
    only contain implementation keys.
    """
    scheme = _lookup(
        sim_config.fvSchemes.laplacianSchemes,
        f"laplacian({field.name})",
        DEFAULT_LAPLACIAN_SCHEME,
    )
    return LAPLACIAN_ALIASES.get(scheme, scheme)


def search_sn_grad_scheme(
    sim_config: SimulatorConfig, field: CellField
) -> SnGradScheme:
    """Resolve ``snGrad(<field>)``."""
    return _lookup(
        sim_config.fvSchemes.snGradSchemes,
        f"snGrad({field.name})",
        DEFAULT_SN_GRAD_SCHEME,
    )


def search_ddt_scheme(
    sim_config: SimulatorConfig, field: CellField
) -> DdtScheme:
    """Resolve ``ddt(<field>)``; history-dependent startup is numerical."""
    return _lookup(
        sim_config.fvSchemes.ddtSchemes,
        f"ddt({field.name})",
        DEFAULT_DDT_SCHEME,
    )
