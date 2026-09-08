"""Shared configuration and workload for benchmarks and profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

from gridfoam.algorithms.factory import create_algorithm
from gridfoam.core.grid.factory import create_grid
from gridfoam.io.vtu import save_export_fields_as_vtu
from gridfoam.meta.config import GridfoamConfig, ManualAlgorithm

PROFILE_DIR = Path(__file__).resolve().parent
REPO_DIR = PROFILE_DIR.parents[1]
DEFAULT_CONFIG = PROFILE_DIR / "data" / "config.yaml"
Resolution = tuple[int, int, int]
StageObserver = Callable[[str], None]


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return result


def parse_resolution(value: str) -> Resolution:
    try:
        parts = tuple(int(v) for v in value.replace(",", " ").split())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use three positive integers") from exc
    if len(parts) != 3 or min(parts) < 1:
        raise argparse.ArgumentTypeError("Use three positive integers")
    return parts


def add_case_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--steps", type=positive_int)
    duration.add_argument("--end-time", type=float)
    parser.add_argument("--cpu-threads", type=positive_int)


def case_arguments(args: argparse.Namespace) -> list[str]:
    """Forward common options to a fresh measurement process."""
    command = ["--config", str(args.config.resolve())]
    for option in ("steps", "end_time", "cpu_threads"):
        value = getattr(args, option)
        if value is not None:
            command.extend(["--" + option.replace("_", "-"), str(value)])
    return command


def available_devices(requested: list[str]) -> list[str]:
    devices = list(dict.fromkeys(requested))
    if "cuda" in devices and not torch.cuda.is_available():
        if devices == ["cuda"]:
            raise RuntimeError("CUDA was requested but is not available")
        print("CUDA is unavailable; running CPU only.", flush=True)
        devices.remove("cuda")
    return devices


def configure_device(device: str, cpu_threads: int | None) -> None:
    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.cuda.init()


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def load_case(
    path: Path,
    device: str,
    *,
    resolution: Resolution | None = None,
    steps: int | None = None,
    end_time: float | None = None,
) -> GridfoamConfig:
    data = yaml.safe_load(path.read_text())
    data["simulator"]["device"] = device
    mesh_path = data["fluxel"].get("mesh_path")
    if mesh_path and not Path(mesh_path).is_absolute():
        data["fluxel"]["mesh_path"] = str(REPO_DIR / mesh_path)
    if resolution is not None:
        data["fluxel"]["root_resolution"] = list(resolution)
    control = data["simulator"]["control"]
    if steps is not None:
        control["endTime"] = steps * control["deltaT"]
    elif end_time is not None:
        control["endTime"] = end_time
    config = GridfoamConfig.model_validate(data)
    if config.simulator.control.endTime <= 0:
        raise ValueError("endTime must be positive")
    if (
        int(config.simulator.control.endTime / config.simulator.control.deltaT)
        < 1
    ):
        raise ValueError("The case must contain at least one step")
    if isinstance(config.simulator.fvSolution.algorithm, ManualAlgorithm):
        raise ValueError("Manual algorithm has no step sequence")
    return config


def config_id(config: GridfoamConfig) -> str:
    data = config.model_dump(mode="json")
    data["simulator"]["control"].pop("output", None)
    return hashlib.sha256(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()[:16]


@dataclass(frozen=True)
class CaseResult:
    device: str
    n_cells: int
    n_steps: int
    setup_s: float
    solve_s: float
    write_s: float
    device_name: str
    cpu_threads: int
    torch_version: str

    @property
    def elapsed_s(self) -> float:
        return self.setup_s + self.solve_s + self.write_s

    def to_dict(self) -> dict[str, str | int | float]:
        return {**asdict(self), "elapsed_s": self.elapsed_s}


def run_case(
    config: GridfoamConfig,
    *,
    observer: StageObserver | None = None,
    fields_path: Path | None = None,
) -> CaseResult:
    """Measure setup and a fixed number of steps, synchronizing CUDA boundaries.

    All tools use this workload. Imports and CUDA context setup happen before
    measurement. Early stopping and post-processing are deliberately excluded.
    VTU export is optional and timed separately.
    """
    device = config.simulator.device.value
    control = config.simulator.control
    n_steps = int(control.endTime / control.deltaT)
    synchronize(device)
    start = time.perf_counter()
    grid = create_grid(config)
    algorithm = create_algorithm(grid)
    synchronize(device)
    setup_s = time.perf_counter() - start
    if observer is not None:
        observer("setup")

    start = time.perf_counter()
    for step in range(1, n_steps + 1):
        algorithm.step()
        if observer is not None:
            observer(f"step_{step}")
    synchronize(device)
    solve_s = time.perf_counter() - start

    write_s = 0.0
    if fields_path is not None:
        fields_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        save_export_fields_as_vtu(grid, str(fields_path))
        synchronize(device)
        write_s = time.perf_counter() - start
        if observer is not None:
            observer("export")

    return CaseResult(
        device=device,
        n_cells=grid.num_cells,
        n_steps=n_steps,
        setup_s=setup_s,
        solve_s=solve_s,
        write_s=write_s,
        device_name=(
            torch.cuda.get_device_name()
            if device == "cuda"
            else platform.processor() or platform.machine()
        ),
        cpu_threads=torch.get_num_threads(),
        torch_version=torch.__version__,
    )
