"""
Configuration normalizes scheme keys and values and retains
solver/diagnostic defaults.
"""

from __future__ import annotations

from typing import cast

from gridfoam.meta.config import (
    ContinuityErrorConfig,
    PostProcessingConfig,
    ResidualControlEntry,
    SIMPLEAlgorithm,
    SolverConfig,
    SolverInfoConfig,
    fvSchemesConfig,
    fvSolutionConfig,
)
from gridfoam.meta.enums import (
    AlgorithmType,
    DivScheme,
    GradScheme,
    PreconditionerType,
    SolverType,
)


def test_fvschemes_regularizes_div_scheme_keys() -> None:
    """Extra whitespace around div-scheme keys is stripped on validation."""
    cfg = fvSchemesConfig(
        divSchemes={
            "default  ,  linear": DivScheme.LINEAR,
        },
    )
    assert cfg.divSchemes is not None
    assert "default, linear" in cfg.divSchemes
    assert cfg.divSchemes["default, linear"] is DivScheme.LINEAR


def test_fvschemes_accepts_grad_scheme_values() -> None:
    """Grad-scheme values may be supplied as strings and coerced to enums."""
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


def test_solver_config_defaults() -> None:
    """
    Solver defaults are no preconditioner, tolerance 1e-6, and a
    1000-iteration limit.
    """
    sc = SolverConfig(method=SolverType.CG)
    assert sc.preconditioner is PreconditionerType.NONE
    assert sc.tolerance == 1e-6
    assert sc.max_iter == 1000


def test_residual_control_entry_and_adjust_phi_defaults() -> None:
    """
    Residual settings retain scalar/object entries and flux adjustment
    defaults to enabled.
    """
    entry = ResidualControlEntry(tolerance=1e-6, rel_tolerance=0.01)
    assert entry.tolerance == 1e-6
    assert entry.rel_tolerance == 0.01

    algo = SIMPLEAlgorithm(
        type=AlgorithmType.SIMPLE,
        residualControl={
            "p": 1e-8,
            "U": ResidualControlEntry(tolerance=1e-6, rel_tolerance=0.1),
        },
        consistent=True,
    )
    assert algo.consistent is True
    assert algo.residualControl["p"] == 1e-8

    fv = fvSolutionConfig(
        algorithm=algo,
        solvers={"p": SolverConfig(method=SolverType.CG)},
    )
    assert fv.adjustPhi is True


def test_post_processing_diagnostics_config_parses() -> None:
    """
    Diagnostics retain default phi, requested solver fields, and the output
    interval.
    """
    cfg = PostProcessingConfig(
        continuityError=ContinuityErrorConfig(),
        solverInfo=SolverInfoConfig(fields=["U", "p"], writeInterval=2),
    )
    assert cfg.continuityError is not None
    assert cfg.continuityError.phi == "phi"
    assert cfg.solverInfo is not None
    assert cfg.solverInfo.fields == ["U", "p"]
    assert cfg.solverInfo.writeInterval == 2
