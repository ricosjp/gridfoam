"""Replayable, fixed-step PISO/PIMPLE maps at physical time boundaries."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from gridfoam.algorithms.pimple import PIMPLE
from gridfoam.algorithms.piso import PISO
from gridfoam.core.state import TensorState, validate_history
from gridfoam.meta.config import PIMPLEAlgorithm, PISOAlgorithm
from gridfoam.models.turbulence.laminar import Laminar
from gridfoam.optimize._inference import call_without_inference
from gridfoam.optimize._replay import AlgorithmReplay, DesignApplicator

type TransientAlgorithm = PISO | PIMPLE
type TransientDesignApplicator = DesignApplicator[TransientAlgorithm]


@dataclass(frozen=True)
class TransientState:
    """Numerical state and non-differentiable physical-time metadata.

    ``values`` contains U, p, packed phi, rAU, rAtU, HbyA, phiHbyA,
    U/old, U/older, phi/old and phi/older. Flux histories contain only
    single-sided faces, as used by ddtCorr. Older blocks always exist to
    keep tensor layouts fixed; they are ignored when previous_dt is None.
    That startup state makes backward (BDF2) use Euler for its first step.

    Use with_values to replace differentiable blocks, clone to isolate
    storage without detaching, and checkpoint only for detached snapshots.
    Time and step_index describe the boundary before the next step.
    """

    values: TensorState
    time: float
    step_index: int
    delta_t: float
    previous_dt: float | None

    def __post_init__(self) -> None:
        if not math.isfinite(self.time):
            raise ValueError("State time must be finite")
        if type(self.step_index) is not int or self.step_index < 0:
            raise ValueError("step_index must be a nonnegative integer")
        if not math.isfinite(self.delta_t) or self.delta_t <= 0:
            raise ValueError("delta_t must be finite and positive")
        if self.previous_dt is not None and self.previous_dt != self.delta_t:
            raise ValueError("Fixed-step history must use delta_t")

    def with_values(self, values: TensorState) -> TransientState:
        """Replace tensor blocks, preserving layout and time metadata."""
        self.values.validate_layout(values)
        return replace(self, values=values)

    def clone(self) -> TransientState:
        """Copy tensor storage while retaining its autograd graph."""
        return replace(self, values=self.values.clone())

    def checkpoint(self) -> TransientState:
        """Make an independent snapshot without an autograd graph."""
        return replace(self, values=self.values.checkpoint())


class TransientStepMap:
    """Evaluate one complete physical time step with explicit history.

    The initial scope is fixed geometry, laminar flow, fixed deltaT and
    corrector counts, and first derivatives. PIMPLE residualControl must be
    empty. Internal correctors are not DAG connection points. Field and
    history state is owned by gridfoam; the caller applies all variable
    boundary/model inputs through apply_design and restores them on exit.

    Calls sharing an algorithm must be sequential. Geometry, configuration,
    registry and numerical controls must not change during the map's life.
    Diagnostics and the caller's ambient state are restored even on failure.
    Primal and implicit transpose solve failures raise LinearSolveError.
    """

    def __init__(
        self,
        algorithm: TransientAlgorithm,
        apply_design: TransientDesignApplicator,
        *,
        initial_time: float = 0.0,
        initial_step: int = 0,
    ):
        if type(algorithm) not in (PISO, PIMPLE):
            raise TypeError("TransientStepMap requires PISO or PIMPLE")
        if type(algorithm.turbulence) is not Laminar:
            raise ValueError("TransientStepMap currently supports laminar flow")
        self.algorithm = algorithm
        self.apply_design = apply_design
        self._replay = AlgorithmReplay(algorithm)
        self._controls = self._read_controls()
        self.initial_state = self._read_state(
            initial_time, initial_step
        ).checkpoint()

    def _read_controls(self) -> tuple[object, ...]:
        algorithm = self.algorithm
        config = algorithm.grid.sim_config.fvSolution.algorithm
        if isinstance(algorithm, PIMPLE):
            if not isinstance(config, PIMPLEAlgorithm):
                raise ValueError(
                    "PIMPLE step map requires PIMPLE configuration"
                )
            if config.residualControl or algorithm.residual_control_enabled:
                raise ValueError(
                    "PIMPLE step maps require empty residualControl"
                )
            if algorithm.n_outer_correctors != config.nOuterCorrectors:
                raise ValueError(
                    "Fixed step controls differ from configuration"
                )
        elif not isinstance(config, PISOAlgorithm):
            raise ValueError("PISO step map requires PISO configuration")
        if (
            algorithm.n_correctors != config.nCorrectors
            or algorithm.n_non_orthogonal_correctors
            != config.nNonOrthogonalCorrectors
        ):
            raise ValueError("Fixed step controls differ from configuration")
        return (
            algorithm.grid.dt,
            algorithm.n_correctors,
            algorithm.n_non_orthogonal_correctors,
            algorithm.n_outer_correctors
            if isinstance(algorithm, PIMPLE)
            else None,
            algorithm.consistent,
            algorithm.adjust_phi_enabled,
            algorithm.p_needs_ref,
            getattr(algorithm, "p_ref_cell", None),
            getattr(algorithm, "p_ref_value", None),
        )

    def _validate_controls(self) -> None:
        if self._read_controls() != self._controls:
            raise ValueError("Fixed step controls have changed")
        self._replay.baseline.validate(self.algorithm)

    def _read_state(self, time: float, step_index: int) -> TransientState:
        algorithm = self.algorithm
        values = dict(self._replay.read_current())
        if algorithm.U.previous_dt != algorithm.phi.previous_dt:
            raise ValueError("U and phi history time steps must agree")
        for name, old, older in (
            ("U", algorithm.U.old_data, algorithm.U.older_data),
            (
                "phi",
                algorithm.phi.old_single_data,
                algorithm.phi.older_single_data,
            ),
        ):
            if (older is None) != (algorithm.U.previous_dt is None):
                raise ValueError("U and phi history validity must agree")
            values[f"{name}/old"] = old
            values[f"{name}/older"] = old if older is None else older
        return TransientState(
            TensorState(values).clone(),
            time,
            step_index,
            algorithm.grid.dt,
            algorithm.U.previous_dt,
        )

    def _validate_state(self, state: TransientState) -> None:
        self.initial_state.values.validate_layout(state.values)
        if state.delta_t != self.initial_state.delta_t:
            raise ValueError("State delta_t differs from the fixed time step")
        for name in ("U", "phi"):
            reference = state.values[name]
            if name == "phi":
                reference = reference[: self.algorithm.phi.num_single_sided]
            validate_history(
                reference,
                state.values[f"{name}/old"],
                state.values[f"{name}/older"]
                if state.previous_dt is not None
                else None,
                state.previous_dt,
            )

    def _write_state(self, state: TransientState) -> None:
        values = state.values.clone()
        self._replay.write_current(values)
        for name, field in (
            ("U", self.algorithm.U),
            ("phi", self.algorithm.phi),
        ):
            field.restore_history(
                values[f"{name}/old"],
                values[f"{name}/older"]
                if state.previous_dt is not None
                else None,
                state.previous_dt,
            )

    def step(
        self, state: TransientState, design: TensorState
    ) -> TransientState:
        """Return the next boundary state, without advancing ambient fields."""
        if torch.is_inference_mode_enabled():
            return call_without_inference(
                lambda: self.step(state.checkpoint(), design.checkpoint())
            )
        self._validate_controls()
        self._validate_state(state)
        with self._replay.restore_ambient():
            self._write_state(state)
            with self._replay.evaluation(self.apply_design, design):
                self._validate_controls()
                self.algorithm.step()
                return self._read_state(
                    state.time + state.delta_t, state.step_index + 1
                )


class PisoStepMap(TransientStepMap):
    """PISO physical-time step map; see TransientStepMap for the contract."""

    def __init__(
        self,
        algorithm: PISO,
        apply_design: TransientDesignApplicator,
        *,
        initial_time: float = 0.0,
        initial_step: int = 0,
    ):
        if type(algorithm) is not PISO:
            raise TypeError("PisoStepMap requires PISO")
        super().__init__(
            algorithm,
            apply_design,
            initial_time=initial_time,
            initial_step=initial_step,
        )


class PimpleStepMap(TransientStepMap):
    """PIMPLE physical-time step map with residual-based exit disabled."""

    def __init__(
        self,
        algorithm: PIMPLE,
        apply_design: TransientDesignApplicator,
        *,
        initial_time: float = 0.0,
        initial_step: int = 0,
    ):
        if type(algorithm) is not PIMPLE:
            raise TypeError("PimpleStepMap requires PIMPLE")
        super().__init__(
            algorithm,
            apply_design,
            initial_time=initial_time,
            initial_step=initial_step,
        )
