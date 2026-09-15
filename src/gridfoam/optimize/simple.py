"""Replayable SIMPLE step map on a fixed grid with explicit design inputs."""

import torch

from gridfoam.algorithms.simple import SIMPLE
from gridfoam.core.state import TensorState
from gridfoam.models.turbulence.laminar import Laminar
from gridfoam.optimize._inference import call_without_inference
from gridfoam.optimize._replay import AlgorithmReplay, DesignApplicator


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

    def __init__(
        self, algorithm: SIMPLE, apply_design: DesignApplicator[SIMPLE]
    ):
        if type(algorithm) is not SIMPLE:
            raise TypeError("SimpleStepMap requires SIMPLE")
        if type(algorithm.turbulence) is not Laminar:
            raise ValueError("SimpleStepMap currently supports laminar flow")
        self.algorithm = algorithm
        self.apply_design = apply_design
        self._replay = AlgorithmReplay(algorithm)
        self.initial_state = self._replay.read_current().checkpoint()

    def step(self, state: TensorState, design: TensorState) -> TensorState:
        """Evaluate one step, retaining links to both input tensor mappings."""
        if torch.is_inference_mode_enabled():
            return call_without_inference(
                lambda: self.step(state.checkpoint(), design.checkpoint())
            )
        self.initial_state.validate_layout(state)
        with self._replay.restore_ambient():
            self._replay.write_current(state, reset_cell_history=True)
            with self._replay.evaluation(self.apply_design, design):
                self.algorithm.step()
                return self._replay.read_current()
