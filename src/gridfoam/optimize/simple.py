"""Replayable SIMPLE step map on a fixed grid with explicit design inputs."""

from collections.abc import Callable
from contextlib import AbstractContextManager

import torch

from gridfoam.algorithms.checkpoint import AlgorithmCheckpoint
from gridfoam.algorithms.simple import SIMPLE
from gridfoam.core.field import CellField, FaceField
from gridfoam.core.state import TensorState
from gridfoam.models.turbulence.laminar import Laminar

DesignApplicator = Callable[[SIMPLE, TensorState], AbstractContextManager[None]]


class SimpleStepMap:
    """Evaluate ``G(state, design)`` without advancing the caller's fields.

    The state includes U, p, packed phi and the four SIMPLE auxiliary fields.
    Fixed-flux pressure boundaries can read previous auxiliary values during
    the momentum predictor, so they cannot all be discarded as scratch.
    Histories, diagnostic counters and unrelated fields belong to the replay
    checkpoint instead. Geometry, configuration and solver settings stay fixed.

    ``apply_design`` must install every variable input on entry and restore it
    on exit, including on failure. It may accept named NN outputs as well as
    optimization parameters. It must not alter geometry or the field registry.
    Calls on the same algorithm must be sequential; higher-order derivatives
    and non-laminar models are outside the initial contract.
    """

    def __init__(self, algorithm: SIMPLE, apply_design: DesignApplicator):
        if type(algorithm) is not SIMPLE:
            raise TypeError("SimpleStepMap requires SIMPLE")
        if type(algorithm.turbulence) is not Laminar:
            raise ValueError("SimpleStepMap currently supports laminar flow")
        self.algorithm = algorithm
        self.apply_design = apply_design
        self._fields: dict[str, CellField | FaceField] = {
            "U": algorithm.U,
            "p": algorithm.p,
            "phi": algorithm.phi,
            "rAU": algorithm.rAU,
            "rAtU": algorithm.rAtU,
            "HbyA": algorithm.HbyA,
            "phiHbyA": algorithm.phi_hbya,
        }
        self._baseline = AlgorithmCheckpoint.capture(algorithm)
        self.initial_state = self._read_state().checkpoint()

    def _read_state(self) -> TensorState:
        return TensorState(
            {
                name: field.data
                if isinstance(field, CellField)
                else field.pack()
                for name, field in self._fields.items()
            }
        ).clone()

    def step(self, state: TensorState, design: TensorState) -> TensorState:
        """Evaluate one step, retaining links to both input tensor mappings."""
        if torch.is_inference_mode_enabled():
            # FV cache tokens require normal tensors with version counters.
            with torch.inference_mode(False), torch.no_grad():
                return self.step(state.checkpoint(), design.checkpoint())
        self.initial_state.validate_layout(state)
        ambient = AlgorithmCheckpoint.capture(self.algorithm, detach=False)
        # Validate baseline before entering a context that might change inputs.
        self._baseline.restore(self.algorithm)
        try:
            for name, field in self._fields.items():
                value = state[name].clone()
                if isinstance(field, CellField):
                    field.data = value
                    field.restore_history(value, None, None)
                else:
                    field.replace_packed(value)
            with (
                self.algorithm.suspend_diagnostics(),
                self.apply_design(self.algorithm, design),
            ):
                self.algorithm.grid.invalidate_derived_caches()
                self.algorithm.step()
                return self._read_state()
        finally:
            ambient.restore(self.algorithm)
