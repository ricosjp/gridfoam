from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "gridfoam" / "outputs" / "sweep_summary.csv"
OF_SUMMARY_PATH = ROOT / "of" / "outputs" / "sweep_summary.csv"


def _wang_piecewise_cd(re_value: float) -> tuple[float, float]:
    # Wang et al. (Appl. Sci. 2025, Eq. (8), Table 4)
    if 0.4 < re_value < 10.0:
        p1, p2, p3, p4, p5 = -0.02, -3.633, 3.907, 0.754, -0.378
        cd = 24.0 / re_value * (1.0 + p1 * re_value**p2) + (
            p3 * re_value**p4 / (re_value + p5)
        )
        return cd, cd
    if 10.0 < re_value < 1.0e3:
        # lower bound (alpha=0 deg), upper bound (alpha=45 deg)
        p_lo = (0.249, 0.586, 0.06, 1.416, 818.921)
        p_hi = (0.213, 0.653, 0.05, 1.344, 152.85)
        cd_lo = 24.0 / re_value * (1.0 + p_lo[0] * re_value ** p_lo[1]) + (
            p_lo[2] * re_value ** p_lo[3] / (re_value + p_lo[4])
        )
        cd_hi = 24.0 / re_value * (1.0 + p_hi[0] * re_value ** p_hi[1]) + (
            p_hi[2] * re_value ** p_hi[3] / (re_value + p_hi[4])
        )
        return cd_lo, cd_hi
    if 1.0e3 < re_value < 4.516e3:
        p1, p2, p3, p4, p5 = 0.338, 0.145, 0.669, 1.044, -0.002
        cd = 24.0 / re_value * (1.0 + p1 * re_value**p2) + (
            p3 * re_value**p4 / (re_value + p5)
        )
        return cd, cd
    raise ValueError(
        "Re is out of Wang et al. cube-correlation validity range."
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
    print("cube drag comparison (gridfoam / OpenFOAM / Wang et al. 2025)")
    print()
    print(
        f"{'Re_eq':>10} {'Cd(gridfoam)':>14} {'Cd(OpenFOAM)':>14} "
        f"{'Cd(paper)':>16}"
    )
    print("-" * 72)
    gf_map = dict(gridfoam_rows)
    of_map = dict(openfoam_rows)

    for re_value in sorted(set(gf_map) | set(of_map)):
        try:
            cd_lo, cd_hi = _wang_piecewise_cd(re_value)
        except ValueError:
            cd_gridfoam = gf_map.get(re_value)
            cd_openfoam = of_map.get(re_value)
            gf_text = "-" if cd_gridfoam is None else f"{cd_gridfoam:.6g}"
            of_text = "-" if cd_openfoam is None else f"{cd_openfoam:.6g}"
            print(f"{re_value:10.4g} {gf_text:>14} {of_text:>14} {'-':>16}")
            continue
        cd_gridfoam = gf_map.get(re_value)
        cd_openfoam = of_map.get(re_value)
        gf_text = "-" if cd_gridfoam is None else f"{cd_gridfoam:.6g}"
        of_text = "-" if cd_openfoam is None else f"{cd_openfoam:.6g}"
        if abs(cd_hi - cd_lo) < 1.0e-12:
            paper_text = f"{cd_lo:16.6g}"
            print(f"{re_value:10.4g} {gf_text:>14} {of_text:>14} {paper_text}")
        else:
            print(
                f"{re_value:10.4g} {gf_text:>14} {of_text:>14} "
                f"{f'[{cd_lo:.4g}, {cd_hi:.4g}]':>16}"
            )
    return


if __name__ == "__main__":
    main()
