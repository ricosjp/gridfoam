from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "gridfoam" / "outputs" / "sweep_summary.csv"
OF_SUMMARY_PATH = ROOT / "of" / "outputs" / "sweep_summary.csv"


def _clift_cd(re: float) -> float:
    # Clift et al. (1978)
    w = math.log(re)
    if re < 0.01:
        return (24.0 / re) * (1.0 + (3.0/16.0) * re)
    if re < 20.0:
        return (24.0 / re) * (1.0 + 0.1315 * re**(0.82-0.05*w))
    if re < 260.0:
        return (24.0 / re) * (1.0 + 0.1935 * re**(0.6305))

    raise ValueError(
        "Re is out of validity range."
    )


def _read_gridfoam_rows() -> list[tuple[float, float]]:
    if not SUMMARY_PATH.exists():
        return []
    with SUMMARY_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [(float(row["Re"]), float(row["Cd"])) for row in rows]


def _read_openfoam_rows() -> list[tuple[float, float]]:
    if not OF_SUMMARY_PATH.exists():
        return []
    with OF_SUMMARY_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [(float(row["Re"]), float(row["Cd"])) for row in rows]


def main() -> None:
    gridfoam_rows = _read_gridfoam_rows()
    openfoam_rows = _read_openfoam_rows()

    print("sphere drag comparison (gridfoam / OpenFOAM / Cheng 2009)")
    print()
    print(
        f"{'Re':>10} {'Cd(gridfoam)':>14} "
        f"{'Cd(OpenFOAM)':>14} {'Cd(paper)':>12}"
    )
    print("-" * 60)
    gf_map = dict(gridfoam_rows)
    of_map = dict(openfoam_rows)

    for re_value in sorted(set(gf_map) | set(of_map)):
        cd_paper = _clift_cd(re_value)
        cd_gridfoam = gf_map.get(re_value)
        cd_openfoam = of_map.get(re_value)
        gf_text = "-" if cd_gridfoam is None else f"{cd_gridfoam:.6g}"
        of_text = "-" if cd_openfoam is None else f"{cd_openfoam:.6g}"
        print(
            f"{re_value:10.4g} {gf_text:>14} {of_text:>14} {cd_paper:12.6g}"
        )
    return


if __name__ == "__main__":
    main()

