"""Device-aware benchmark CSV storage, including legacy OpenFOAM results."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from tests.profile.case import PROFILE_DIR, Resolution

RESULTS_DIR = PROFILE_DIR / "benchmark_results"
RESULTS_CSV = RESULTS_DIR / "resolution_scaling.csv"
CsvRow = dict[str, str]
CSV_FIELDS = [
    "solver",
    "device",
    "resolution",
    "nx",
    "ny",
    "nz",
    "n_cells",
    "elapsed_s",
    "setup_s",
    "solve_s",
    "end_time",
    "delta_t",
    "n_steps",
    "time_per_step_s",
    "config_id",
    "cpu_threads",
    "device_name",
    "torch_version",
]


@dataclass(frozen=True)
class BenchmarkRow:
    solver: str
    device: str
    resolution: Resolution
    n_cells: int
    elapsed_s: float
    end_time: float
    delta_t: float
    setup_s: float | None = None
    solve_s: float | None = None
    config_id: str = ""
    cpu_threads: int | None = None
    device_name: str = ""
    torch_version: str = ""

    def to_csv_row(self) -> CsvRow:
        nx, ny, nz = self.resolution
        n_steps = int(self.end_time / self.delta_t)
        values = {
            **asdict(self),
            "resolution": f"{nx}x{ny}x{nz}",
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "n_steps": n_steps,
            "time_per_step_s": self.elapsed_s / max(n_steps, 1),
        }
        return {
            key: "" if values[key] is None else str(values[key])
            for key in CSV_FIELDS
        }


def load_rows(path: Path) -> list[CsvRow]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        # Never guess the device of the old combined gridfoam CSV.
        if not row.get("device"):
            row["device"] = "cpu" if row["solver"] == "openfoam" else "unknown"
            if row["solver"] == "gridfoam":
                for device in ("cpu", "cuda"):
                    if path.name.startswith(device + "-"):
                        row["device"] = device
    return rows


def row_key(row: CsvRow) -> tuple[str, str, str, float, float, str, str]:
    return (
        row["solver"],
        row["device"],
        row["resolution"],
        float(row["end_time"]),
        float(row["delta_t"]),
        row.get("config_id", ""),
        row.get("cpu_threads", ""),
    )


def upsert_rows(path: Path, rows: list[BenchmarkRow]) -> None:
    if not rows:
        return
    merged = {row_key(row): row for row in load_rows(path)}
    for result in rows:
        row = result.to_csv_row()
        merged[row_key(row)] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(merged.values())
    temporary.replace(path)


def import_cached_openfoam(path: Path) -> None:
    """Migrate only OpenFOAM rows; gridfoam device provenance may be unknown."""
    known = {row_key(row) for row in load_rows(path)}
    for source in (
        RESULTS_CSV,
        RESULTS_DIR / "cpu-resolution_scaling.csv",
        RESULTS_DIR / "cuda-resolution_scaling.csv",
    ):
        if source.resolve() == path.resolve():
            continue
        imported = []
        for row in load_rows(source):
            if row["solver"] != "openfoam" or row_key(row) in known:
                continue
            imported.append(
                BenchmarkRow(
                    solver="openfoam",
                    device="cpu",
                    resolution=(int(row["nx"]), int(row["ny"]), int(row["nz"])),
                    n_cells=int(row["n_cells"]),
                    elapsed_s=float(row["elapsed_s"]),
                    end_time=float(row["end_time"]),
                    delta_t=float(row["delta_t"]),
                )
            )
            known.add(row_key(row))
        upsert_rows(path, imported)
