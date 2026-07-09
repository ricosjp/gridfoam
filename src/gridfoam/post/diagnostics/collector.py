"""Streaming CSV diagnostics collector for continuityError and solverInfo."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TextIO

from gridfoam.core.field import FaceField, get_or_create_facefield
from gridfoam.core.grid.base import IGridBase
from gridfoam.core.name import make_field_name
from gridfoam.meta.config import PostProcessingConfig
from gridfoam.meta.enums import FieldRole
from gridfoam.post.diagnostics.continuity import compute_continuity_error
from gridfoam.solvers.base import SolveStats


@dataclass(frozen=True)
class SolverInfoColumn:
    """One solverInfo CSV column group for a field component."""

    csv_key: str
    field_name: str
    component_index: int


class DiagnosticsCollector:
    """
    Stream post-processing diagnostics to CSV files during simulation.

    Each enabled diagnostic appends one row per recorded step and flushes
    immediately so memory use stays constant.
    """

    def __init__(
        self,
        *,
        grid: IGridBase,
        config: PostProcessingConfig,
        output_dir: Path,
        n_steps: int,
        delta_t: float,
        phase: str | None = None,
    ) -> None:
        self._grid = grid
        self._config = config
        self._output_dir = output_dir
        self._n_steps = n_steps
        self._delta_t = delta_t
        self._phase = phase
        self._cumulative_continuity = 0.0
        self._closed = False

        self._continuity_file: TextIO | None = None
        self._continuity_writer = None
        self._solver_info_file: TextIO | None = None
        self._solver_info_writer = None
        self._solver_info_columns: list[SolverInfoColumn] = []

        output_dir.mkdir(parents=True, exist_ok=True)

        if config.continuityError is not None:
            path = output_dir / "continuity_error.csv"
            self._continuity_file = path.open("w", newline="")
            self._continuity_writer = csv.writer(self._continuity_file)
            self._continuity_writer.writerow(
                ["time", "local", "global", "cumulative"]
            )
            self._continuity_file.flush()

        if config.solverInfo is not None:
            self._solver_info_columns = self._build_solver_info_columns(config)
            path = output_dir / "solver_info.csv"
            self._solver_info_file = path.open("w", newline="")
            self._solver_info_writer = csv.writer(self._solver_info_file)
            header = ["time"]
            for column in self._solver_info_columns:
                header.extend(
                    [
                        f"{column.csv_key}_solver",
                        f"{column.csv_key}_initial",
                        f"{column.csv_key}_final",
                        f"{column.csv_key}_iters",
                        f"{column.csv_key}_converged",
                    ]
                )
            self._solver_info_writer.writerow(header)
            self._solver_info_file.flush()

    @classmethod
    def from_config(
        cls,
        grid: IGridBase,
        post_processing: PostProcessingConfig | None,
        output_dir: Path,
        *,
        n_steps: int,
        delta_t: float,
        phase: str | None = None,
    ) -> DiagnosticsCollector | None:
        """
        Create a collector when any diagnostic post-processing is enabled.

        Parameters
        ----------
        grid : IGridBase
            Computational grid.
        post_processing : PostProcessingConfig | None
            Post-processing configuration.
        output_dir : Path
            Directory for CSV output files.
        n_steps : int
            Total number of algorithm steps.
        delta_t : float
            Time step size for converting step index to simulation time.
        phase : str | None, optional
            Optional multiphase field prefix.

        Returns
        -------
        DiagnosticsCollector | None
            Collector instance, or ``None`` when diagnostics are disabled.
        """
        if post_processing is None:
            return None
        if (
            post_processing.continuityError is None
            and post_processing.solverInfo is None
        ):
            return None
        return cls(
            grid=grid,
            config=post_processing,
            output_dir=output_dir,
            n_steps=n_steps,
            delta_t=delta_t,
            phase=phase,
        )

    def _resolve_solver_info_fields(
        self,
        config: PostProcessingConfig,
    ) -> list[str]:
        assert config.solverInfo is not None
        if config.solverInfo.fields is not None:
            return list(config.solverInfo.fields)
        return ["U", "p"]

    def _resolve_field_num_components(self, field_name: str) -> int:
        field = self._grid.get_cellfield(field_name)
        if field is not None:
            return field.num_components

        condition = self._grid.sim_config.get_field_condition(field_name)
        if condition is not None and condition.internal:
            return len(condition.internal)

        return 1

    def _build_solver_info_columns(
        self,
        config: PostProcessingConfig,
    ) -> list[SolverInfoColumn]:
        columns: list[SolverInfoColumn] = []
        for field_name in self._resolve_solver_info_fields(config):
            num_components = self._resolve_field_num_components(field_name)
            if num_components == 1:
                columns.append(
                    SolverInfoColumn(
                        csv_key=field_name,
                        field_name=field_name,
                        component_index=0,
                    )
                )
                continue
            for component_index in range(num_components):
                columns.append(
                    SolverInfoColumn(
                        csv_key=f"{field_name}_{component_index}",
                        field_name=field_name,
                        component_index=component_index,
                    )
                )
        return columns

    def _should_record(self, step: int, write_interval: int) -> bool:
        return step % write_interval == 0 or step == self._n_steps

    def record_step(
        self,
        step: int,
        phi: FaceField,
        solve_stats: dict[str, tuple[SolveStats, ...]],
    ) -> None:
        """
        Append diagnostic rows for the current step when intervals match.

        Parameters
        ----------
        step : int
            Current algorithm step (1-based).
        phi : FaceField
            Face flux field after pressure/flux correction.
        solve_stats : dict[str, tuple[SolveStats, ...]]
            Per-component solver statistics keyed by field name.
        """
        if self._closed:
            return

        simulation_time = step * self._delta_t
        continuity_cfg = self._config.continuityError
        if (
            continuity_cfg is not None
            and self._continuity_writer is not None
            and self._should_record(step, continuity_cfg.writeInterval)
        ):
            local_error, global_error = compute_continuity_error(
                phi,
                self._grid.cell_volumes,
            )
            self._cumulative_continuity += global_error
            self._continuity_writer.writerow(
                [
                    simulation_time,
                    local_error,
                    global_error,
                    self._cumulative_continuity,
                ]
            )
            assert self._continuity_file is not None
            self._continuity_file.flush()

        solver_info_cfg = self._config.solverInfo
        if (
            solver_info_cfg is not None
            and self._solver_info_writer is not None
            and self._should_record(step, solver_info_cfg.writeInterval)
        ):
            row: list[float | str | int] = [simulation_time]
            for column in self._solver_info_columns:
                stats_tuple = solve_stats.get(column.field_name)
                if stats_tuple is None or column.component_index >= len(
                    stats_tuple
                ):
                    row.extend(["", "", "", "", ""])
                    continue
                stats = stats_tuple[column.component_index]
                row.extend(
                    [
                        stats.solver,
                        stats.initial_residual,
                        stats.final_residual,
                        stats.iterations,
                        int(stats.converged),
                    ]
                )
            self._solver_info_writer.writerow(row)
            assert self._solver_info_file is not None
            self._solver_info_file.flush()

    def get_phi_field(self) -> FaceField:
        """
        Return the configured flux field for continuity error evaluation.

        Returns
        -------
        FaceField
            Flux field referenced by ``continuityError.phi``.
        """
        assert self._config.continuityError is not None
        phi_name = make_field_name(
            self._config.continuityError.phi,
            phase=self._phase,
        )
        return get_or_create_facefield(
            self._grid,
            phi_name,
            FieldRole.LOCAL,
            1,
        )

    def close(self) -> None:
        """Flush and close all open CSV file handles."""
        if self._closed:
            return
        if self._continuity_file is not None:
            self._continuity_file.flush()
            self._continuity_file.close()
            self._continuity_file = None
            self._continuity_writer = None
        if self._solver_info_file is not None:
            self._solver_info_file.flush()
            self._solver_info_file.close()
            self._solver_info_file = None
            self._solver_info_writer = None
        self._closed = True

    def __enter__(self) -> DiagnosticsCollector:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
