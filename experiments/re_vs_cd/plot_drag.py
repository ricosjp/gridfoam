from __future__ import annotations

import argparse
import csv
import math
import re
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = ROOT / "outputs"
PARAMETERS_PATH = ROOT / "data" / "parameters.yml"
DEFAULT_OUTPUT = OUTPUTS_ROOT / "re_vs_cd_comparison.png"

RE_DIR_PATTERN = re.compile(r"^re_(?P<re>.+)$")


class CaseConfig(BaseModel):
    name: str
    mesh_path: str
    magU_ref: float
    A_ref: float
    L_ref: float


class ExperimentParameters(BaseModel):
    case: list[CaseConfig]
    Re: list[float]


def _load_parameters(path: Path) -> ExperimentParameters:
    with path.open() as f:
        return ExperimentParameters.model_validate(yaml.safe_load(f))


def _parse_re_dir(name: str) -> float | None:
    match = RE_DIR_PATTERN.match(name)
    if match is None:
        return None
    return float(match.group("re"))


def _read_gridfoam_rows(case_name: str) -> list[tuple[float, float]]:
    case_dir = OUTPUTS_ROOT / case_name / "gridfoam"
    if not case_dir.exists():
        return []

    rows: list[tuple[float, float]] = []
    for re_dir in sorted(case_dir.iterdir()):
        if not re_dir.is_dir():
            continue
        re_value = _parse_re_dir(re_dir.name)
        if re_value is None:
            continue

        coeffs_path = re_dir / "force_coeffs.csv"
        if not coeffs_path.exists():
            continue

        with coeffs_path.open(newline="") as f:
            reader = csv.DictReader(f)
            coeffs = list(reader)
        if not coeffs:
            continue

        rows.append((re_value, float(coeffs[-1]["Cd"])))
    return rows


def _read_openfoam_numeric_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                rows.append([float(item) for item in stripped.split()])
            except ValueError:
                continue
    return rows


def _read_openfoam_rows(case_name: str) -> list[tuple[float, float]]:
    case_dir = OUTPUTS_ROOT / case_name / "openfoam"
    if not case_dir.exists():
        return []

    rows: list[tuple[float, float]] = []
    for re_dir in sorted(case_dir.iterdir()):
        if not re_dir.is_dir():
            continue
        re_value = _parse_re_dir(re_dir.name)
        if re_value is None:
            continue

        force_coeffs_dir = re_dir / "postProcessing" / "forceCoeffs1" / "0"
        candidates = sorted(force_coeffs_dir.glob("**/coefficient.dat"))
        if not candidates:
            continue

        coeffs = _read_openfoam_numeric_rows(candidates[-1])
        if not coeffs:
            continue

        rows.append((re_value, coeffs[-1][1]))
    return rows


def _clift_cd(re: float) -> float:
    # Clift et al. (1978).
    w = math.log(re)
    if re < 0.01:
        return (24.0 / re) * (1.0 + (3.0 / 16.0) * re)
    if re < 20.0:
        return (24.0 / re) * (1.0 + 0.1315 * re ** (0.82 - 0.05 * w))
    if re < 260.0:
        return (24.0 / re) * (1.0 + 0.1935 * re**0.6305)

    raise ValueError("Re is out of validity range.")


def _sucker_brauwer_cd(re: float) -> float:
    # Sucker-Brauwer (1975).
    return (
        6.8 / re**0.89
        + 1.96 / re**0.5
        + 1.18
        - 0.0004 * re / (1 + 3.63e-7 * re * re)
    )


def _logspace(start: float, stop: float, count: int) -> list[float]:
    if start <= 0.0 or stop <= 0.0:
        raise ValueError("Log-space bounds must be positive.")
    if count < 2:
        return [start]

    log_start = math.log10(start)
    log_stop = math.log10(stop)
    return [
        10 ** (log_start + (log_stop - log_start) * i / (count - 1))
        for i in range(count)
    ]


def _plot_rows(
    ax: plt.Axes,
    rows: list[tuple[float, float]],
    *,
    label: str,
    marker: str,
) -> None:
    if not rows:
        return

    re_values, cd_values = zip(*sorted(rows), strict=True)
    ax.loglog(re_values, cd_values, marker=marker, linewidth=1.5, label=label)


def _plot_correlation(
    ax: plt.Axes,
    re_values: list[float],
    *,
    label: str,
    cd_function: Callable[[float], float],
) -> None:
    if not re_values:
        return

    xs = _logspace(min(re_values), max(re_values), 160)
    ys = [cd_function(re) for re in xs]
    ax.loglog(xs, ys, linestyle="--", color="black", linewidth=1.2, label=label)


CASE_METADATA: dict[str, dict[str, object]] = {
    "sphere": {
        "title": "Sphere",
        "correlation_label": "Clift et al.",
        "cd_function": _clift_cd,
    },
    "cylinder": {
        "title": "Circular cylinder",
        "correlation_label": "Sucker-Brauwer",
        "cd_function": _sucker_brauwer_cd,
    },
    "cube": {
        "title": "Cube",
        "correlation_label": None,
        "cd_function": None,
    },
}


def _plot_case(ax: plt.Axes, case_name: str) -> None:
    metadata = CASE_METADATA.get(
        case_name,
        {"title": case_name, "correlation_label": None, "cd_function": None},
    )
    title = str(metadata["title"])
    correlation_label = metadata["correlation_label"]
    cd_function = metadata["cd_function"]

    gridfoam_rows = _read_gridfoam_rows(case_name)
    openfoam_rows = _read_openfoam_rows(case_name)
    _plot_rows(ax, gridfoam_rows, label="gridfoam", marker="o")
    _plot_rows(ax, openfoam_rows, label="OpenFOAM", marker="s")

    if correlation_label is not None and cd_function is not None:
        all_re = [re for re, _ in gridfoam_rows + openfoam_rows]
        _plot_correlation(
            ax,
            all_re,
            label=str(correlation_label),
            cd_function=cd_function,
        )

    ax.set_title(title)
    ax.set_xlabel("Re")
    ax.set_ylabel("Cd")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend()


def plot(output: Path, cases: list[str]) -> None:
    n_cases = len(cases)
    fig, axes = plt.subplots(
        1,
        n_cases,
        figsize=(4.3 * n_cases, 4),
        constrained_layout=True,
        squeeze=False,
    )

    for ax, case_name in zip(axes[0], cases, strict=True):
        _plot_case(ax, case_name)

    fig.suptitle("Low-Reynolds Drag Comparison")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parameters = _load_parameters(PARAMETERS_PATH)
    case_names = [case.name for case in parameters.case]

    parser = argparse.ArgumentParser(
        description=(
            "Plot low-Reynolds drag comparisons from "
            "experiments/re_vs_cd outputs."
        )
    )
    parser.add_argument(
        "--cases",
        choices=case_names,
        nargs="+",
        default=case_names,
        help="Cases to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output image path. Default: {DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()

    plot(args.output, args.cases)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
