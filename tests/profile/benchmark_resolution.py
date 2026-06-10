"""
Sweep root mesh resolution and record cell count vs elapsed time.

OpenFOAM: patch ``blockMeshDict``, run ``Allclean`` + ``Allrun``, read
``elapsed_time.txt``, and count cells with ``checkMesh``.

gridfoam: vary ``root_resolution`` in ``data/config.yml``, time mesh
generation plus the SIMPLE loop, and read ``grid.num_cells``.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

PROFILE_DIR = Path(__file__).resolve().parent
OF_CASE_DIR = PROFILE_DIR / "of"
GRIDFOAM_CONFIG = PROFILE_DIR / "data" / "config.yml"
RESULTS_DIR = PROFILE_DIR / "benchmark_results"
RESULTS_CSV = RESULTS_DIR / "resolution_scaling.csv"

RESOLUTIONS: list[tuple[int, int, int]] = [
    (5, 2, 2),
    (10, 4, 4),
    (20, 8, 8),
    (40, 16, 16),
]

BLOCK_MESH_RE = re.compile(
    r"hex \(0 1 2 3 4 5 6 7\) \(\d+ \d+ \d+\) simpleGrading"
)
CHECKMESH_CELLS_RE = re.compile(r"^\s*cells:\s+(\d+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class BenchmarkRow:
    solver: str
    resolution: tuple[int, int, int]
    n_cells: int
    elapsed_s: float
    end_time: float
    delta_t: float

    @property
    def resolution_label(self) -> str:
        nx, ny, nz = self.resolution
        return f"{nx}x{ny}x{nz}"

    def to_csv_row(self) -> dict[str, str | int | float]:
        nx, ny, nz = self.resolution
        n_steps = int(self.end_time / self.delta_t)
        return {
            "solver": self.solver,
            "resolution": self.resolution_label,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "n_cells": self.n_cells,
            "elapsed_s": f"{self.elapsed_s:.3f}",
            "end_time": self.end_time,
            "delta_t": self.delta_t,
            "n_steps": n_steps,
            "time_per_step_s": f"{self.elapsed_s / max(n_steps, 1):.6f}",
        }


def _resolution_key(resolution: tuple[int, int, int]) -> str:
    return "x".join(str(v) for v in resolution)


def _load_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _append_rows(path: Path, rows: list[BenchmarkRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "solver",
        "resolution",
        "nx",
        "ny",
        "nz",
        "n_cells",
        "elapsed_s",
        "end_time",
        "delta_t",
        "n_steps",
        "time_per_step_s",
    ]
    existing = _load_existing_rows(path)
    existing_keys = {
        (row["solver"], row["resolution"]) for row in existing if row
    }
    new_rows = [
        row.to_csv_row()
        for row in rows
        if (row.solver, row.resolution_label) not in existing_keys
    ]
    if not new_rows:
        return

    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header and not existing:
            writer.writeheader()
        writer.writerows(new_rows)


def _patch_block_mesh_dict(path: Path, resolution: tuple[int, int, int]) -> str:
    text = path.read_text()
    nx, ny, nz = resolution
    replacement = f"hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading"
    new_text, count = BLOCK_MESH_RE.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Failed to patch blockMeshDict at {path}")
    path.write_text(new_text)
    return text


def _patch_control_dict_end_time(path: Path, end_time: float) -> str:
    text = path.read_text()
    new_text, count = re.subn(
        r"(?m)^endTime\s+\d+(?:\.\d+)?;\s*$",
        f"endTime         {end_time:g};",
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"Failed to patch endTime in {path}")
    path.write_text(new_text)
    return text


def _count_openfoam_cells(case_dir: Path) -> int:
    result = subprocess.run(
        ["checkMesh", "-constant"],
        cwd=case_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    match = CHECKMESH_CELLS_RE.search(result.stdout)
    if match is None:
        raise RuntimeError("checkMesh output did not contain cell count")
    return int(match.group(1))


def _read_openfoam_timing(case_dir: Path) -> float:
    elapsed_path = case_dir / "elapsed_time.txt"
    if not elapsed_path.exists():
        raise FileNotFoundError(
            f"Missing {elapsed_path}. Run Allrun before collecting results."
        )
    return float(elapsed_path.read_text().strip())


def _read_openfoam_control(case_dir: Path) -> tuple[float, float]:
    control = case_dir / "system" / "controlDict"
    text = control.read_text()
    end_match = re.search(r"(?m)^endTime\s+(\d+(?:\.\d+)?);\s*$", text)
    dt_match = re.search(r"(?m)^deltaT\s+(\d+(?:\.\d+)?);\s*$", text)
    if end_match is None or dt_match is None:
        raise RuntimeError(f"Failed to parse controlDict at {control}")
    return float(end_match.group(1)), float(dt_match.group(1))


def run_openfoam_benchmark(
    resolutions: list[tuple[int, int, int]],
    *,
    end_time: float | None,
    skip_existing: bool,
) -> list[BenchmarkRow]:
    if shutil.which("blockMesh") is None:
        raise RuntimeError("OpenFOAM not found in PATH")

    block_mesh_dict = OF_CASE_DIR / "system" / "blockMeshDict"
    control_dict = OF_CASE_DIR / "system" / "controlDict"
    original_block_mesh = block_mesh_dict.read_text()
    original_control = control_dict.read_text()

    rows: list[BenchmarkRow] = []
    existing = _load_existing_rows(RESULTS_CSV)
    existing_keys = {(row["solver"], row["resolution"]) for row in existing}

    try:
        if end_time is not None:
            _patch_control_dict_end_time(control_dict, end_time)

        for resolution in resolutions:
            label = "x".join(str(v) for v in resolution)
            if skip_existing and ("openfoam", label) in existing_keys:
                print(f"[openfoam] skip existing {label}")
                continue

            print(f"[openfoam] running resolution={resolution}")
            block_mesh_dict.write_text(original_block_mesh)
            _patch_block_mesh_dict(block_mesh_dict, resolution)

            subprocess.run(["./Allclean"], cwd=OF_CASE_DIR, check=True)
            subprocess.run(["./Allrun"], cwd=OF_CASE_DIR, check=True)

            elapsed_s = _read_openfoam_timing(OF_CASE_DIR)
            n_cells = _count_openfoam_cells(OF_CASE_DIR)
            case_end_time, delta_t = _read_openfoam_control(OF_CASE_DIR)
            rows.append(
                BenchmarkRow(
                    solver="openfoam",
                    resolution=resolution,
                    n_cells=n_cells,
                    elapsed_s=elapsed_s,
                    end_time=case_end_time,
                    delta_t=delta_t,
                )
            )
            print(
                f"[openfoam] {label}: n_cells={n_cells}, elapsed={elapsed_s:.3f}s"
            )
    finally:
        block_mesh_dict.write_text(original_block_mesh)
        control_dict.write_text(original_control)

    _append_rows(RESULTS_CSV, rows)
    return rows


def run_gridfoam_benchmark(
    resolutions: list[tuple[int, int, int]],
    *,
    end_time: float | None,
    device: str | None,
    skip_existing: bool,
) -> list[BenchmarkRow]:
    from gridfoam.algorithms.simple import SIMPLE
    from gridfoam.boundaries.factory import apply_boundary_condition_configs
    from gridfoam.core.field import CellField
    from gridfoam.core.grid.factory import create_grid
    from gridfoam.meta.config import GridfoamConfig
    from gridfoam.meta.enums import FieldRole
    from gridfoam.models.turbulence.laminar import Laminar

    with GRIDFOAM_CONFIG.open() as f:
        base_config = yaml.safe_load(f)

    rows: list[BenchmarkRow] = []
    existing = _load_existing_rows(RESULTS_CSV)
    existing_keys = {(row["solver"], row["resolution"]) for row in existing}

    for resolution in resolutions:
        label = "x".join(str(v) for v in resolution)
        if skip_existing and ("gridfoam", label) in existing_keys:
            print(f"[gridfoam] skip existing {label}")
            continue

        config_data = yaml.safe_load(GRIDFOAM_CONFIG.read_text())
        config_data["fluxel"]["root_resolution"] = list(resolution)
        if end_time is not None:
            config_data["simulator"]["control"]["endTime"] = end_time
        if device is not None:
            config_data["simulator"]["device"] = device

        config = GridfoamConfig.model_validate(config_data)
        delta_t = config.simulator.control.deltaT
        case_end_time = config.simulator.control.endTime
        n_steps = int(case_end_time / delta_t)

        print(f"[gridfoam] running resolution={resolution}, steps={n_steps}")
        start = time.perf_counter()

        grid = create_grid(config)
        n_cells = grid.num_cells

        U = CellField(grid, "U", role=FieldRole.LOCAL, num_components=3)
        p = CellField(grid, "p", role=FieldRole.LOCAL, num_components=1)
        boundary_conditions = config.simulator.boundaryConditions
        if boundary_conditions is None:
            raise ValueError("boundaryConditions is required")
        apply_boundary_condition_configs(U, boundary_conditions["U"])
        apply_boundary_condition_configs(p, boundary_conditions["p"])

        turbulence = Laminar(grid=grid, nu=0.1)
        algo = SIMPLE(grid=grid, U=U, p=p, turbulence=turbulence)
        for _ in range(n_steps):
            algo.step()

        elapsed_s = time.perf_counter() - start
        rows.append(
            BenchmarkRow(
                solver="gridfoam",
                resolution=resolution,
                n_cells=n_cells,
                elapsed_s=elapsed_s,
                end_time=case_end_time,
                delta_t=delta_t,
            )
        )
        print(f"[gridfoam] {label}: n_cells={n_cells}, elapsed={elapsed_s:.3f}s")

    _append_rows(RESULTS_CSV, rows)
    return rows


def _parse_resolutions(values: list[str] | None) -> list[tuple[int, int, int]]:
    if not values:
        return RESOLUTIONS

    parsed: list[tuple[int, int, int]] = []
    for value in values:
        parts = value.replace(",", " ").split()
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"Resolution must have 3 integers, got {value!r}"
            )
        parsed.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--resolution",
        action="append",
        metavar="NX NY NZ",
        help="Mesh resolution to run (default: 5x2x2, 10x4x4, 20x8x8, 40x16x16)",
    )
    common.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Override endTime for both solvers to make elapsed time comparable",
    )
    common.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip solver/resolution pairs already present in the CSV",
    )

    openfoam_parser = subparsers.add_parser(
        "openfoam",
        parents=[common],
        help="Run OpenFOAM resolution sweep",
    )
    openfoam_parser.set_defaults(func=lambda args: run_openfoam_benchmark(
        _parse_resolutions(args.resolution),
        end_time=args.end_time,
        skip_existing=args.skip_existing,
    ))

    gridfoam_parser = subparsers.add_parser(
        "gridfoam",
        parents=[common],
        help="Run gridfoam resolution sweep",
    )
    gridfoam_parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Override simulator.device in config.yml",
    )
    gridfoam_parser.set_defaults(func=lambda args: run_gridfoam_benchmark(
        _parse_resolutions(args.resolution),
        end_time=args.end_time,
        device=args.device,
        skip_existing=args.skip_existing,
    ))

    args = parser.parse_args(argv)
    args.func(args)
    print(f"Results appended to {RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
