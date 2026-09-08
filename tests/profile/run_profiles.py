"""Run CPU/CUDA workloads sequentially with separate profile outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tests.profile.case import (
    PROFILE_DIR,
    REPO_DIR,
    add_case_arguments,
    available_devices,
    case_arguments,
    parse_resolution,
    positive_int,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["time", "memory"])
    add_case_arguments(parser)
    parser.add_argument(
        "--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"]
    )
    parser.add_argument("--resolution", type=parse_resolution)
    parser.add_argument(
        "--output", type=Path, help="Defaults to outputs/time or outputs/memory"
    )
    parser.add_argument("--write-fields", action="store_true")
    parser.add_argument(
        "--native", action="store_true", help="Include Memray C/C++ stacks"
    )
    parser.add_argument("--history-entries", type=positive_int, default=100_000)
    args = parser.parse_args()
    output = (args.output or PROFILE_DIR / "outputs" / args.mode).resolve()
    output.mkdir(parents=True, exist_ok=True)
    devices = available_devices(args.devices)
    summaries = {}
    for device in devices:
        directory = output / device
        command = [
            sys.executable,
            "-m",
            "tests.profile.profile_case",
            args.mode,
            "--device",
            device,
            "--output",
            str(directory),
            "--history-entries",
            str(args.history_entries),
            *case_arguments(args),
        ]
        if args.resolution is not None:
            command.extend(
                ["--resolution", " ".join(map(str, args.resolution))]
            )
        if args.write_fields:
            command.append("--write-fields")
        if args.native:
            command.append("--native")
        print(f"Profiling {args.mode} on {device}", flush=True)
        subprocess.run(command, cwd=REPO_DIR, check=True)
        summaries[device] = json.loads((directory / "summary.json").read_text())
    # A skipped device is absent, even if an older per-device report exists.
    (output / "summary.json").write_text(
        json.dumps(
            {
                "requested_devices": args.devices,
                "completed_devices": devices,
                "results": summaries,
            },
            indent=2,
        )
    )
    print(f"Comparison summary: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
