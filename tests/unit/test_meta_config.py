"""Unit tests for ``gridfoam.meta.config`` validation helpers."""

from __future__ import annotations

from typing import cast

from gridfoam.meta.config import SolverConfig, fvSchemesConfig
from gridfoam.meta.enums import (
    DivScheme,
    GradScheme,
    PreconditionerType,
    SolverType,
)


def test_fvschemes_regularizes_div_scheme_keys():
    cfg = fvSchemesConfig(
        divSchemes={
            "default  ,  linear": DivScheme.LINEAR,
        },
    )
    assert cfg.divSchemes is not None
    assert "default, linear" in cfg.divSchemes
    assert cfg.divSchemes["default, linear"] is DivScheme.LINEAR


def test_fvschemes_accepts_grad_scheme_values():
    cfg = fvSchemesConfig(
        gradSchemes=cast(
            "dict[str, GradScheme]",
            {
                "default": "linear",
                "grad(p)": "leastsquare",
            },
        ),
    )

    assert cfg.gradSchemes is not None
    assert cfg.gradSchemes["default"] is GradScheme.LINEAR
    assert cfg.gradSchemes["grad(p)"] is GradScheme.LEASTSQUARE


def test_solver_config_defaults():
    sc = SolverConfig(method=SolverType.CG)
    assert sc.preconditioner is PreconditionerType.NONE
    assert sc.tolerance == 1e-6
    assert sc.max_iter == 1000
