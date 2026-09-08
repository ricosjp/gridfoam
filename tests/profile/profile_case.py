"""Profile one case/device in a fresh process; called by run_profiles."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import memray
import torch
from pyinstrument import Profiler
from tests.profile.case import (
    CaseResult,
    add_case_arguments,
    config_id,
    configure_device,
    load_case,
    parse_resolution,
    run_case,
    synchronize,
)

from gridfoam.meta.config import GridfoamConfig


class CudaMemoryRecorder:
    """Record CUDA counters and bounded history at stage boundaries."""

    def __init__(self, directory: Path, n_steps: int, history_entries: int):
        self.directory = directory
        self.n_steps = n_steps
        self.history_entries = history_entries
        self.samples: list[dict[str, str | float | int]] = []
        self.started = time.perf_counter()

    def start(self) -> None:
        torch.cuda.reset_peak_memory_stats()
        # PyTorch's documented memory-snapshot API uses underscored names.
        torch.cuda.memory._record_memory_history(  # pyright: ignore[reportPrivateUsage]
            max_entries=self.history_entries,
            stacks="python",
        )
        self.record("start")

    def stop(self) -> None:
        torch.cuda.memory._record_memory_history(  # pyright: ignore[reportPrivateUsage]
            enabled=None
        )

    def record(self, stage: str) -> None:
        torch.cuda.synchronize()
        self.samples.append(
            {
                "stage": stage,
                "elapsed_s": time.perf_counter() - self.started,
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
        # Keep initialization even if the bounded history later wraps.
        if stage == "setup" or stage == f"step_{self.n_steps}":
            name = "initialization" if stage == "setup" else "final"
            torch.cuda.memory._dump_snapshot(  # pyright: ignore[reportPrivateUsage]
                str(self.directory / f"cuda-{name}.pickle")
            )


def write_memory_plot(
    directory: Path,
    host_samples: list[Any],
    cuda_samples: list[dict[str, Any]],
) -> None:
    count = 2 if cuda_samples else 1
    fig, axes = plt.subplots(count, 1, figsize=(9, 4 * count), squeeze=False)
    ax = axes[0, 0]
    if host_samples:
        start = host_samples[0].time
        times = [(s.time - start) / 1000 for s in host_samples]
        ax.plot(
            times, [s.rss / 2**20 for s in host_samples], label="Process RSS"
        )
        ax.plot(
            times,
            [s.heap / 2**20 for s in host_samples],
            label="Memray tracked heap",
        )
    ax.set_title("Host memory (not GPU memory)")
    if cuda_samples:
        ax = axes[1, 0]
        times = [float(s["elapsed_s"]) for s in cuda_samples]
        for key, label in (
            ("allocated_bytes", "Allocated at stage end"),
            ("reserved_bytes", "Reserved by allocator"),
            ("peak_allocated_bytes", "Peak allocated so far"),
        ):
            ax.plot(
                times, [int(s[key]) / 2**20 for s in cuda_samples], label=label
            )
        ax.set_title("CUDA memory (PyTorch allocator)")
    for ax in axes[:, 0]:
        ax.set_xlabel("Elapsed time [s]")
        ax.set_ylabel("MiB")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "memory.png", dpi=160)
    plt.close(fig)


def profile_memory(
    args: argparse.Namespace, config: GridfoamConfig
) -> tuple[CaseResult, dict[str, Any]]:
    directory = args.output
    capture = directory / "host.bin"
    cuda = (
        CudaMemoryRecorder(
            directory,
            int(
                config.simulator.control.endTime
                / config.simulator.control.deltaT
            ),
            args.history_entries,
        )
        if args.device == "cuda"
        else None
    )
    result: CaseResult | None = None
    try:
        if cuda is not None:
            cuda.start()
        with memray.Tracker(
            destination=memray.FileDestination(str(capture), overwrite=True),
            native_traces=args.native,
        ):
            result = run_case(
                config,
                observer=cuda.record if cuda is not None else None,
                fields_path=directory / "fields.vtu"
                if args.write_fields
                else None,
            )
            synchronize(args.device)
    finally:
        if cuda is not None:
            cuda.stop()

    reader = memray.FileReader(str(capture))
    host_samples = list(reader.get_memory_snapshots())
    cuda_samples = cuda.samples if cuda is not None else []
    memory: dict[str, Any] = {
        "host_peak_tracked_bytes": reader.metadata.peak_memory,
        "host_peak_rss_bytes": max((s.rss for s in host_samples), default=0),
        "native_traces": args.native,
        "cuda_samples": cuda_samples,
    }
    if cuda is not None:
        memory["cuda_peak_allocated_bytes"] = max(
            int(s["peak_allocated_bytes"]) for s in cuda.samples
        )
        memory["cuda_peak_reserved_bytes"] = max(
            int(s["peak_reserved_bytes"]) for s in cuda.samples
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "memray",
            "flamegraph",
            "-f",
            "-o",
            str(directory / "host.html"),
            str(capture),
        ],
        check=True,
    )
    write_memory_plot(directory, host_samples, cuda_samples)
    assert result is not None
    return result, memory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["time", "memory"])
    add_case_arguments(parser)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--resolution", type=parse_resolution)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-fields", action="store_true")
    parser.add_argument(
        "--native", action="store_true", help="Include Memray C/C++ stacks"
    )
    parser.add_argument("--history-entries", type=int, default=100_000)
    args = parser.parse_args()
    if args.history_entries < 1:
        parser.error("--history-entries must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    configure_device(args.device, args.cpu_threads)
    config = load_case(
        args.config,
        args.device,
        resolution=args.resolution,
        steps=args.steps,
        end_time=args.end_time,
    )
    # Store the exact effective configuration alongside every profile.
    (args.output / "config.json").write_text(config.model_dump_json(indent=2))
    if args.mode == "time":
        profiler = Profiler()
        with profiler:
            result = run_case(
                config,
                fields_path=args.output / "fields.vtu"
                if args.write_fields
                else None,
            )
        profiler.write_html(args.output / "profile.html")
        extra: dict[str, Any] = {}
    else:
        result, extra = profile_memory(args, config)
    summary = {**result.to_dict(), "config_id": config_id(config), **extra}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{args.device}] {args.mode}: {args.output}", flush=True)


if __name__ == "__main__":
    main()
