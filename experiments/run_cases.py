import resource
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from experiments.schema import CompareConfig

of_bashrc = "/usr/lib/openfoam/openfoam2412/etc/bashrc"


@dataclass(frozen=True)
class RunTiming:
    wall_time_s: float
    cpu_user_s: float
    cpu_system_s: float

    @property
    def cpu_total_s(self) -> float:
        return self.cpu_user_s + self.cpu_system_s


def _child_cpu_usage() -> resource.struct_rusage:
    return resource.getrusage(resource.RUSAGE_CHILDREN)


def _format_timing(label: str, timing: RunTiming) -> str:
    return (
        f"{label}: "
        f"wall={timing.wall_time_s:.3f}s, "
        f"cpu={timing.cpu_total_s:.3f}s "
        f"(user={timing.cpu_user_s:.3f}s, sys={timing.cpu_system_s:.3f}s)"
    )


def _run_timed(
    label: str,
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
) -> RunTiming:
    start_wall = time.perf_counter()
    start_cpu = _child_cpu_usage()
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except BaseException:
        end_cpu = _child_cpu_usage()
        timing = RunTiming(
            wall_time_s=time.perf_counter() - start_wall,
            cpu_user_s=end_cpu.ru_utime - start_cpu.ru_utime,
            cpu_system_s=end_cpu.ru_stime - start_cpu.ru_stime,
        )
        print(_format_timing(label, timing) + " [failed]")
        raise

    end_cpu = _child_cpu_usage()
    timing = RunTiming(
        wall_time_s=time.perf_counter() - start_wall,
        cpu_user_s=end_cpu.ru_utime - start_cpu.ru_utime,
        cpu_system_s=end_cpu.ru_stime - start_cpu.ru_stime,
    )
    print(_format_timing(label, timing))
    return timing


def run_openfoam(cfg: CompareConfig) -> RunTiming | None:
    if not cfg.openfoam.enabled:
        return None
    of_case = cfg.case_dir / "of"
    if not of_case.is_dir():
        msg = f"OpenFOAM case_dir not found: {of_case}"
        raise FileNotFoundError(msg)
    return _run_timed(
        "openfoam",
        ["bash", "-lc", f"source {of_bashrc} && ./Allclean && ./Allrun"],
        cwd=of_case,
    )


def run_gridfoam(cfg: CompareConfig) -> RunTiming | None:
    if not cfg.gridfoam.enabled:
        return None
    gridfoam_script = cfg.case_dir / "gridfoam" / cfg.gridfoam.script
    if not gridfoam_script.is_file():
        msg = f"gridfoam script not found: {gridfoam_script}"
        raise FileNotFoundError(msg)
    return _run_timed(
        "gridfoam",
        ["uv", "run", "python", str(gridfoam_script)],
    )
