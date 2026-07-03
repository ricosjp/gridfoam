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
    # Extra whitespace around div-scheme keys is stripped on validation.
    cfg = fvSchemesConfig(
        divSchemes={
            "default  ,  linear": DivScheme.LINEAR,
        },
    )
    assert cfg.divSchemes is not None
    assert "default, linear" in cfg.divSchemes
    assert cfg.divSchemes["default, linear"] is DivScheme.LINEAR


def test_fvschemes_accepts_grad_scheme_values():
    # Grad-scheme values may be supplied as strings and coerced to enums.
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
    # Default solver settings match OpenFOAM-style baseline values.
    sc = SolverConfig(method=SolverType.CG)
    assert sc.preconditioner is PreconditionerType.NONE
    assert sc.tolerance == 1e-6
    assert sc.max_iter == 1000


def test_residual_control_entry_and_adjust_phi_defaults():
    # Residual-control entries and fvSolution defaults parse correctly.
    from gridfoam.meta.config import (
        ResidualControlEntry,
        SIMPLEAlgorithm,
        fvSolutionConfig,
    )
    from gridfoam.meta.enums import AlgorithmType

    entry = ResidualControlEntry(tolerance=1e-6, rel_tolerance=0.01)
    assert entry.tolerance == 1e-6
    assert entry.rel_tolerance == 0.01

    algo = SIMPLEAlgorithm(
        type=AlgorithmType.SIMPLE,
        residualControl={"p": 1e-8, "U": {"tolerance": 1e-6, "rel_tolerance": 0.1}},
        consistent=True,
    )
    assert algo.consistent is True
    assert algo.residualControl["p"] == 1e-8

    fv = fvSolutionConfig(
        algorithm=algo,
        solvers={"p": SolverConfig(method=SolverType.CG)},
    )
    assert fv.adjustPhi is True
