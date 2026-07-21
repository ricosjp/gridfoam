"""Parse OpenFOAM simpleFoam logs into gridfoam-compatible diagnostic CSVs."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = ROOT / "outputs"

TIME_PATTERN = re.compile(
    r"^Time\s*=\s*"
    r"(?P<time>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)
SOLVE_PATTERN = re.compile(
    r"^(?P<solver>\w+):\s+Solving for (?P<field>\w+),"
    r"\s+Initial residual = (?P<initial>[^,]+),"
    r"\s+Final residual = (?P<final>[^,]+),"
    r"\s+No Iterations (?P<iters>\d+)\s*$"
)
CONTINUITY_PATTERN = re.compile(
    r"^time step continuity errors\s*:"
    r"\s*sum local = (?P<local>[^,]+),"
    r"\s*global = (?P<global>[^,]+),"
    r"\s*cumulative = (?P<cumulative>\S+)\s*$"
)

# Normalize OpenFOAM field names for diagnostic CSVs.
FIELD_NAME_MAP = {
    "Ux": "Ux",
    "Uy": "Uy",
    "Uz": "Uz",
    "U": "Ux",
    "p": "p",
}

SOLVER_FIELDS = ("Ux", "Uy", "Uz", "p")


@dataclass
class _SolveRecord:
    solver: str
    initial: float
    final: float
    iters: int


@dataclass
class _TimeStep:
    time: float
    solves: dict[str, list[_SolveRecord]] = field(default_factory=dict)
    local: float | None = None
    global_: float | None = None
    cumulative: float | None = None


def _normalize_field_name(name: str) -> str | None:
    return FIELD_NAME_MAP.get(name)


def _aggregate_solves(records: list[_SolveRecord]) -> _SolveRecord:
    """Collapse non-orthogonal pressure corrects into one residual row.

    Uses the first initial residual, the last final residual, and the sum of
    linear-solver iterations within the outer SIMPLE step.
    """
    first = records[0]
    last = records[-1]
    return _SolveRecord(
        solver=first.solver,
        initial=first.initial,
        final=last.final,
        iters=sum(item.iters for item in records),
    )


def parse_simplefoam_log(path: Path) -> list[_TimeStep]:
    """Parse residual and continuity lines from ``log.simpleFoam``."""
    steps: list[_TimeStep] = []
    current: _TimeStep | None = None

    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            time_match = TIME_PATTERN.match(line)
            if time_match is not None:
                if current is not None:
                    steps.append(current)
                current = _TimeStep(time=float(time_match.group("time")))
                continue

            if current is None:
                continue

            solve_match = SOLVE_PATTERN.match(line)
            if solve_match is not None:
                csv_key = _normalize_field_name(solve_match.group("field"))
                if csv_key is None:
                    continue
                record = _SolveRecord(
                    solver=solve_match.group("solver").lower(),
                    initial=float(solve_match.group("initial")),
                    final=float(solve_match.group("final")),
                    iters=int(solve_match.group("iters")),
                )
                current.solves.setdefault(csv_key, []).append(record)
                continue

            continuity_match = CONTINUITY_PATTERN.match(line)
            if continuity_match is not None:
                current.local = float(continuity_match.group("local"))
                current.global_ = float(continuity_match.group("global"))
                current.cumulative = float(continuity_match.group("cumulative"))

    if current is not None:
        steps.append(current)

    return [step for step in steps if step.solves or step.local is not None]


def write_diagnostic_csvs(
    run_dir: Path,
    *,
    log_name: str = "log.simpleFoam",
    overwrite: bool = False,
) -> tuple[Path | None, Path | None]:
    """
    Write ``continuity_error.csv`` and ``solver_info.csv`` under ``run_dir``.

    Parameters
    ----------
    run_dir : Path
        OpenFOAM case directory containing ``log.simpleFoam``.
    log_name : str, optional
        Solver log file name.
    overwrite : bool, optional
        Replace existing CSVs when True.

    Returns
    -------
    tuple[Path | None, Path | None]
        Paths to the written continuity and solver-info CSVs.
    """
    log_path = run_dir / log_name
    if not log_path.exists():
        raise FileNotFoundError(f"OpenFOAM log not found: {log_path}")

    continuity_path = run_dir / "continuity_error.csv"
    solver_path = run_dir / "solver_info.csv"
    if not overwrite and continuity_path.exists() and solver_path.exists():
        return continuity_path, solver_path

    steps = parse_simplefoam_log(log_path)
    if not steps:
        raise ValueError(f"No diagnostic rows parsed from {log_path}")

    continuity_written: Path | None = None
    solver_written: Path | None = None

    continuity_rows = [
        step
        for step in steps
        if step.local is not None
        and step.global_ is not None
        and step.cumulative is not None
    ]
    if continuity_rows and (overwrite or not continuity_path.exists()):
        with continuity_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "local", "global", "cumulative"])
            for step in continuity_rows:
                writer.writerow(
                    [step.time, step.local, step.global_, step.cumulative]
                )
        continuity_written = continuity_path

    solver_rows = [step for step in steps if step.solves]
    if solver_rows and (overwrite or not solver_path.exists()):
        header = ["time"]
        for name in SOLVER_FIELDS:
            header.extend(
                [
                    f"{name}_solver",
                    f"{name}_initial",
                    f"{name}_final",
                    f"{name}_iters",
                    f"{name}_converged",
                ]
            )
        with solver_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for step in solver_rows:
                row: list[object] = [step.time]
                for name in SOLVER_FIELDS:
                    records = step.solves.get(name)
                    if not records:
                        row.extend(["", "", "", "", ""])
                        continue
                    agg = _aggregate_solves(records)
                    row.extend(
                        [
                            agg.solver,
                            agg.initial,
                            agg.final,
                            agg.iters,
                            1,
                        ]
                    )
                writer.writerow(row)
        solver_written = solver_path

    return continuity_written, solver_written


def ensure_openfoam_diagnostic_csvs(
    run_dir: Path,
    *,
    overwrite: bool = False,
) -> bool:
    """Ensure diagnostic CSVs exist for an OpenFOAM run directory."""
    continuity_path = run_dir / "continuity_error.csv"
    solver_path = run_dir / "solver_info.csv"
    log_path = run_dir / "log.simpleFoam"
    if continuity_path.exists() and solver_path.exists() and not overwrite:
        with solver_path.open(newline="") as f:
            fieldnames = next(csv.reader(f), [])
        if "Ux_initial" in fieldnames:
            return True
        overwrite = True
    if not log_path.exists():
        return False
    write_diagnostic_csvs(run_dir, overwrite=overwrite)
    return continuity_path.exists() or solver_path.exists()


def _discover_openfoam_run_dirs(outputs_root: Path) -> list[Path]:
    run_dirs: list[Path] = []
    if not outputs_root.exists():
        return run_dirs
    for case_dir in sorted(outputs_root.iterdir()):
        openfoam_dir = case_dir / "openfoam"
        if not openfoam_dir.is_dir():
            continue
        for re_dir in sorted(openfoam_dir.iterdir()):
            if re_dir.is_dir() and (re_dir / "log.simpleFoam").exists():
                run_dirs.append(re_dir)
    return run_dirs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert OpenFOAM log.simpleFoam residual/continuity output into "
            "gridfoam-compatible CSV files under experiments/re_vs_cd/outputs."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        nargs="*",
        default=None,
        help="Specific OpenFOAM run directories. Default: discover all runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite CSV files even when they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dirs = args.run_dir or _discover_openfoam_run_dirs(OUTPUTS_ROOT)
    if not run_dirs:
        raise SystemExit("No OpenFOAM run directories found.")

    for run_dir in run_dirs:
        continuity_path, solver_path = write_diagnostic_csvs(
            run_dir,
            overwrite=args.overwrite,
        )
        if continuity_path is not None:
            print(continuity_path)
        if solver_path is not None:
            print(solver_path)


if __name__ == "__main__":
    main()
