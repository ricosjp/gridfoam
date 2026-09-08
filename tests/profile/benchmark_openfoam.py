"""OpenFOAM case execution and parsing, separate from gridfoam measurements."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from tests.profile.benchmark_results import (
    BenchmarkRow,
    import_cached_openfoam,
    load_rows,
    upsert_rows,
)
from tests.profile.case import PROFILE_DIR, Resolution

OF_CASE_DIR = PROFILE_DIR / "openfoam"
BLOCK_MESH_RE = re.compile(
    r"hex \(0 1 2 3 4 5 6 7\) \(\d+ \d+ \d+\) simpleGrading"
)
CHECKMESH_CELLS_RE = re.compile(r"^\s*cells:\s+(\d+)\s*$", re.MULTILINE)


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
    resolutions: list[Resolution],
    *,
    end_time: float | None,
    steps: int | None,
    skip_existing: bool,
    csv_path: Path,
    allow_unavailable: bool = False,
) -> None:
    if skip_existing:
        import_cached_openfoam(csv_path)
    existing = load_rows(csv_path)
    cached_rows = [row for row in existing if row["solver"] == "openfoam"]
    # Cached measurements carry their own duration when OpenFOAM is absent.
    control = OF_CASE_DIR / "system" / "controlDict"
    if control.exists():
        case_end_time, delta_t = _read_openfoam_control(OF_CASE_DIR)
    elif cached_rows:
        cached = cached_rows[0]
        case_end_time, delta_t = (
            float(cached["end_time"]),
            float(cached["delta_t"]),
        )
    else:
        if allow_unavailable:
            print(f"OpenFOAM case unavailable; skipping: {OF_CASE_DIR}")
            return
        raise RuntimeError(f"OpenFOAM case not found: {OF_CASE_DIR}")
    requested_end = steps * delta_t if steps is not None else end_time
    if requested_end is not None:
        case_end_time = requested_end
    if case_end_time <= 0 or delta_t <= 0:
        raise ValueError("OpenFOAM endTime and deltaT must be positive")
    pending = []
    for resolution in resolutions:
        label = "x".join(map(str, resolution))
        found = any(
            row["solver"] == "openfoam"
            and row["resolution"] == label
            and float(row["end_time"]) == case_end_time
            and float(row["delta_t"]) == delta_t
            for row in existing
        )
        if skip_existing and found:
            print(f"[openfoam] skip existing {label}", flush=True)
        else:
            pending.append(resolution)
    if not pending:
        return
    block_mesh = OF_CASE_DIR / "system" / "blockMeshDict"
    if not control.exists() or not block_mesh.exists():
        if allow_unavailable:
            print(f"OpenFOAM case unavailable; skipping: {OF_CASE_DIR}")
            return
        raise RuntimeError(f"OpenFOAM case incomplete: {OF_CASE_DIR}")
    if shutil.which("blockMesh") is None:
        if allow_unavailable:
            print("OpenFOAM unavailable; skipping uncached resolutions.")
            return
        raise RuntimeError(
            "OpenFOAM not found in PATH; "
            "use --without-openfoam for gridfoam only"
        )
    original_block_mesh = block_mesh.read_text()
    original_control = control.read_text()
    try:
        _patch_control_dict_end_time(control, case_end_time)
        for resolution in pending:
            print(f"[openfoam] running resolution={resolution}", flush=True)
            block_mesh.write_text(original_block_mesh)
            _patch_block_mesh_dict(block_mesh, resolution)
            subprocess.run(["./Allclean"], cwd=OF_CASE_DIR, check=True)
            subprocess.run(["./Allrun"], cwd=OF_CASE_DIR, check=True)
            row = BenchmarkRow(
                solver="openfoam",
                device="cpu",
                resolution=resolution,
                n_cells=_count_openfoam_cells(OF_CASE_DIR),
                elapsed_s=_read_openfoam_timing(OF_CASE_DIR),
                end_time=case_end_time,
                delta_t=delta_t,
            )
            upsert_rows(csv_path, [row])
            print(
                f"[openfoam] cells={row.n_cells}, elapsed={row.elapsed_s:.3f}s",
                flush=True,
            )
    finally:
        block_mesh.write_text(original_block_mesh)
        control.write_text(original_control)
