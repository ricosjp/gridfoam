from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "gridfoam" / "outputs" / "sweep_summary.csv"
OF_SUMMARY_PATH = ROOT / "of" / "outputs" / "sweep_summary.csv"


def _sucker_brauwer_cd(re: float) -> float:
    return (
        6.8 / re**0.89
        + 1.96 / re**0.5
        + 1.18
        - 0.0004*re / (1 + 3.63e-7*re*re)
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

    print("cylinder drag comparison (gridfoam / OpenFOAM / Sucker-Brauwer)")
    print()
    print(
        f"{'Re_D':>10} {'Cd(gridfoam)':>14} "
        f"{'Cd(OpenFOAM)':>14} {'Cd(paper)':>12}"
    )
    print("-" * 62)
    gf_map = dict(gridfoam_rows)
    of_map = dict(openfoam_rows)

    for re_value in sorted(set(gf_map) | set(of_map)):
        cd_paper = _sucker_brauwer_cd(re_value)
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

