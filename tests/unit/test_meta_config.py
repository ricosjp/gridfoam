"""Unit tests for ``gridfoam.meta.config`` validation helpers."""

from __future__ import annotations

from gridfoam.meta.config import SolverConfig, fvSchemesConfig
from gridfoam.meta.enums import DivScheme, PreconditionerType, SolverType


def test_fvschemes_regularizes_div_scheme_keys():
    cfg = fvSchemesConfig(
        divSchemes={
            "default  ,  linear": DivScheme.LINEAR,
        },
    )
    assert "default, linear" in cfg.divSchemes
    assert cfg.divSchemes["default, linear"] is DivScheme.LINEAR


def test_solver_config_defaults():
    sc = SolverConfig(method=SolverType.CG)
    assert sc.preconditioner is PreconditionerType.NONE
    assert sc.tolerance == 1e-6
    assert sc.max_iter == 1000
