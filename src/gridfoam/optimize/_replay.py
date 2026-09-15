"""Shared replay helpers for segregated SIMPLE, PISO and PIMPLE step maps."""

from collections.abc import Callable, Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import Protocol, runtime_checkable

import torch

from gridfoam.algorithms.checkpoint import (
    AlgorithmCheckpoint,
    CheckpointableAlgorithm,
)
from gridfoam.core.dimensions import DIM_LENGTH, DIM_VOLUME, dim_div
from gridfoam.core.field import (
    CellField,
    FaceField,
    get_or_create_cellfield,
    get_or_create_facefield,
)
from gridfoam.core.field_bindings import FieldBindings
from gridfoam.core.grid.base import GridBase
from gridfoam.core.state import TensorState
from gridfoam.meta.enums import FieldRole
from gridfoam.solvers.base import LinearSolver, require_converged_solves

type DesignApplicator[T] = Callable[
    [T, TensorState], AbstractContextManager[None]
]


@runtime_checkable
class SegregatedAlgorithm(CheckpointableAlgorithm, Protocol):
    """SIMPLE, PISO or PIMPLE with the shared pressure--velocity fields."""

    U: CellField
    p: CellField
    phi: FaceField
    rAU: CellField
    rAtU: CellField
    HbyA: CellField
    phi_hbya: FaceField
    grid: GridBase
    solvers: dict[str, LinearSolver]

    def step(self) -> None: ...

    def suspend_diagnostics(self) -> AbstractContextManager[None]: ...


def _prepare_step_fields(
    p: CellField, phi: FaceField, phi_hbya: FaceField, hbya: CellField
) -> tuple[CellField | FaceField, ...]:
    """Pin FVC outputs used by the pressure--velocity coupling step."""
    return (
        get_or_create_facefield(
            hbya.grid,
            f"{hbya.name}_f",
            FieldRole.LOCAL,
            hbya.component_shape,
            dimension=hbya.dimension,
        ),
        get_or_create_cellfield(
            p.grid,
            f"grad({p.name})",
            p.role,
            p.component_shape + (3,),
            dimension=dim_div(p.dimension, DIM_LENGTH),
        ),
        get_or_create_facefield(
            p.grid,
            f"snGrad({p.name})",
            FieldRole.LOCAL,
            p.component_shape,
            dimension=dim_div(p.dimension, DIM_LENGTH),
        ),
        *(
            get_or_create_cellfield(
                face.grid,
                f"div({face.name})",
                FieldRole.LOCAL,
                face.component_shape,
                dimension=dim_div(face.dimension, DIM_VOLUME),
            )
            for face in (phi, phi_hbya)
        ),
    )


def _segregated_bindings(algorithm: SegregatedAlgorithm) -> FieldBindings:
    """Bind the conservative step-map unknowns on ``algorithm``."""
    return FieldBindings(
        {
            "U": algorithm.U,
            "p": algorithm.p,
            "phi": algorithm.phi,
            "rAU": algorithm.rAU,
            "rAtU": algorithm.rAtU,
            "HbyA": algorithm.HbyA,
            "phiHbyA": algorithm.phi_hbya,
        }
    )


class AlgorithmReplay:
    """Restore ambient state around one frozen, strictly solved algorithm step.

    Scratch FVC fields are prepared before the baseline checkpoint so that a
    later exception traceback cannot register new fields during restore.
    Time histories stay with the caller; they are not a field-binding contract.
    """

    def __init__(self, algorithm: SegregatedAlgorithm):
        self.algorithm = algorithm
        self._bindings = _segregated_bindings(algorithm)
        self._scratch = _prepare_step_fields(
            algorithm.p, algorithm.phi, algorithm.phi_hbya, algorithm.HbyA
        )
        self.baseline = AlgorithmCheckpoint.capture(algorithm)
        self._active = False

    def read_current(self) -> TensorState:
        """Clone current bound fields with live autograd links."""
        return self._bindings.read()

    def write_current(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        reset_cell_history: bool = False,
    ) -> None:
        """Replace bound current values; optionally drop cell time levels.

        Unbound keys such as ``U/old`` are ignored. Face histories are left
        unchanged. Cell history reset aliases ``old_data`` to the written
        buffer, matching a steady step that must not see ambient time levels.
        """
        self._bindings.write(
            {key: values[key] for key in self._bindings.fields}
        )
        if reset_cell_history:
            for field in self._bindings.fields.values():
                if isinstance(field, CellField):
                    field.restore_history(field.data, None, None)

    @contextmanager
    def restore_ambient(self) -> Generator[None]:
        """Restore the baseline, then the caller's ambient state on exit."""
        if self._active:
            raise RuntimeError("Step map does not support reentry")
        algorithm = self.algorithm
        ambient = AlgorithmCheckpoint.capture(algorithm, detach=False)
        self._active = True
        try:
            self.baseline.restore(algorithm)
            yield
        finally:
            try:
                ambient.restore(algorithm)
            finally:
                self._active = False

    @contextmanager
    def evaluation(
        self,
        apply_design: Callable[..., AbstractContextManager[None]],
        design: TensorState,
    ) -> Generator[None]:
        """Freeze the registry, apply design inputs, and require solves."""
        with (
            self.algorithm.grid.freeze_field_registry(),
            self.algorithm.suspend_diagnostics(),
            apply_design(self.algorithm, design),
            require_converged_solves(self.algorithm.solvers.values()),
        ):
            self.algorithm.grid.invalidate_derived_caches()
            yield
