"""Run CPU/CUDA resolution sweeps in isolated processes and plot the results."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch
from tests.profile.benchmark_openfoam import run_openfoam_benchmark
from tests.profile.benchmark_results import (
    CSV_FIELDS,
    RESULTS_CSV,
    BenchmarkRow,
    CsvRow,
    load_rows,
    row_key,
    upsert_rows,
)
from tests.profile.case import (
    REPO_DIR,
    Resolution,
    add_case_arguments,
    available_devices,
    case_arguments,
    config_id,
    configure_device,
    load_case,
    parse_resolution,
    run_case,
)
from tests.profile.plot_cells_vs_time import plot_cells_vs_time

RESOLUTIONS: list[Resolution] = [
    (5, 2, 2),
    (10, 4, 4),
    (20, 8, 8),
    (40, 16, 16),
]


def measure_case(args: argparse.Namespace) -> CsvRow:
    """Worker: measure exactly one resolution/device, then persist it."""
    configure_device(args.device, args.cpu_threads)
    resolution = args.resolution[0]
    config = load_case(
        args.config,
        args.device,
        resolution=resolution,
        steps=args.steps,
        end_time=args.end_time,
    )
    control = config.simulator.control
    identity = BenchmarkRow(
        solver="gridfoam",
        device=args.device,
        resolution=resolution,
        n_cells=0,
        elapsed_s=0,
        end_time=control.endTime,
        delta_t=control.deltaT,
        config_id=config_id(config),
        cpu_threads=torch.get_num_threads(),
    )
    key = row_key(identity.to_csv_row())
    if args.skip_existing:
        for row in load_rows(args.csv):
            if row_key(row) == key:
                print(f"[{args.device}] skip existing {resolution}", flush=True)
                return row
    print(f"[{args.device}] running {resolution}", flush=True)
    result = run_case(config)
    row = replace(
        identity,
        n_cells=result.n_cells,
        elapsed_s=result.elapsed_s,
        setup_s=result.setup_s,
        solve_s=result.solve_s,
        device_name=result.device_name,
        torch_version=result.torch_version,
    )
    upsert_rows(args.csv, [row])
    print(
        f"[{args.device}] cells={result.n_cells}, setup={result.setup_s:.3f}s, "
        f"solve={result.solve_s:.3f}s, total={result.elapsed_s:.3f}s",
        flush=True,
    )
    return row.to_csv_row()


def gridfoam_sweep(args: argparse.Namespace) -> list[CsvRow]:
    requested = [args.device] if args.device else args.devices
    rows = []
    with tempfile.TemporaryDirectory(prefix="gridfoam-benchmark-") as directory:
        result_path = Path(directory) / "result.json"
        for device in available_devices(requested):
            for resolution in args.resolution:
                command = [
                    sys.executable,
                    "-m",
                    "tests.profile.benchmark_resolution",
                    "gridfoam",
                    "--worker-result",
                    str(result_path),
                    "--device",
                    device,
                    "--csv",
                    str(args.csv.resolve()),
                    "--resolution",
                    " ".join(map(str, resolution)),
                    *case_arguments(args),
                ]
                if args.skip_existing:
                    command.append("--skip-existing")
                subprocess.run(command, cwd=REPO_DIR, check=True)
                rows.append(json.loads(result_path.read_text()))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=["all", "gridfoam", "openfoam"],
        default="all",
    )
    add_case_arguments(parser)
    parser.add_argument(
        "--resolution",
        action="append",
        type=parse_resolution,
        metavar='"NX NY NZ"',
    )
    parser.add_argument(
        "--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"]
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], help="Run just one device"
    )
    parser.add_argument("--csv", type=Path, default=RESULTS_CSV)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse matching gridfoam configuration/device/duration results",
    )
    parser.add_argument("--without-openfoam", action="store_true")
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.resolution = args.resolution or RESOLUTIONS
    if args.worker_result is not None:
        args.worker_result.write_text(json.dumps(measure_case(args)))
        return

    if args.command != "gridfoam" and not args.without_openfoam:
        # Preserve the make benchmark behavior: reuse OpenFOAM unless forced
        # via the explicit `openfoam` command / make benchmark-openfoam.
        run_openfoam_benchmark(
            args.resolution,
            end_time=args.end_time,
            steps=args.steps,
            skip_existing=args.command == "all" or args.skip_existing,
            csv_path=args.csv,
            allow_unavailable=args.command == "all",
        )
    if args.command == "openfoam":
        return
    rows = gridfoam_sweep(args)
    # Only plot this invocation's gridfoam data, plus comparable OpenFOAM rows.
    durations = {(float(r["end_time"]), float(r["delta_t"])) for r in rows}
    labels = {r["resolution"] for r in rows}
    rows.extend(
        row
        for row in load_rows(args.csv)
        if row["solver"] == "openfoam"
        and not args.without_openfoam
        and row["resolution"] in labels
        and (float(row["end_time"]), float(row["delta_t"])) in durations
    )
    latest_csv = args.csv.with_name(args.csv.stem + "-latest.csv")
    with latest_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    plot_cells_vs_time(
        latest_csv,
        args.csv.with_name("cells_vs_time.png"),
        use_time_per_step=False,
    )
    print(f"Results: {args.csv}")


if __name__ == "__main__":
    main()
