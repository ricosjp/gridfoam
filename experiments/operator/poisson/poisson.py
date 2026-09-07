import argparse
import csv
import gc
import logging
import sys
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from jinja2 import Template
from pydantic import BaseModel

from gridfoam.core.equation import equation
from gridfoam.core.field import CellField, get_or_create_cellfield
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.grid.base import IGridBase
from gridfoam.core.grid.factory import create_grid as create_grid_from_config
from gridfoam.core.name import make_field_name
from gridfoam.fv import fvm
from gridfoam.io.vtu import save_export_fields_as_vtu, to_unstructured_grid
from gridfoam.meta.config import (
    GridfoamConfig,
)
from gridfoam.meta.enums import (
    FieldRole,
)
from gridfoam.solvers.factory import create_solver

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_TEMPLATE_PATH = SCRIPT_DIR / "config.j2"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
LOG_FILE_NAME = "poisson.log"
METRICS_CSV_NAME = "convergence_metrics.csv"
CONVERGENCE_PLOT_NAME = "linf_vs_dx.png"
DEFAULT_N_LEAF_REFINEMENTS = (1, 2, 3, 4, 5)

logger = logging.getLogger(__name__)


class PoissonMetrics(BaseModel, frozen=True):
    n_leaf_refinement: int
    dx: float
    linf: float


def configure_run_logger(log_file: Path) -> None:
    formatter = logging.Formatter("%(message)s")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def load_config(n_leaf_refinement: int) -> GridfoamConfig:
    """Render ``config.j2`` for the requested leaf-refinement level."""
    template = Template(CONFIG_TEMPLATE_PATH.read_text())
    rendered = template.render(n_leaf_refinement=n_leaf_refinement)
    return GridfoamConfig.model_validate(yaml.safe_load(rendered))


def p_exact(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Analytical solution on [-1, 1]^3 with homogeneous Dirichlet boundaries.

    p(x, y, z) = sin(pi/2 * (x+1)) * sin(pi/2 * (y+1)) * sin(pi/2 * (z+1))
    """

    def theta(coord: torch.Tensor) -> torch.Tensor:
        return 0.5 * pi * (coord + 1.0)

    return torch.sin(theta(x)) * torch.sin(theta(y)) * torch.sin(theta(z))


def source_term(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """
    Volumetric source for div(grad(p)) = source.

    source = nabla^2 p_exact = -(3 pi^2 / 4) * p_exact
    """
    return -(3.0 * pi * pi / 4.0) * p_exact(x, y, z)


def assemble_poisson_matrix(p: CellField) -> FvMatrix:
    """Assemble the Poisson system matrix with the manufactured source term."""
    grid = p.grid
    cell_centers = grid.cell_centers
    x = cell_centers[:, 0]
    y = cell_centers[:, 1]
    z = cell_centers[:, 2]

    poisson_mat = fvm.laplacian(1.0, p)
    poisson_mat.source += source_term(x, y, z) * grid.cell_volumes
    return poisson_mat


def compute_errors(
    p_num: torch.Tensor, p_ref: torch.Tensor
) -> dict[str, float]:
    """
    Compute error norms between numerical and reference solutions.

    Parameters
    ----------
    p_num : torch.Tensor
        Numerical solution with shape ``[C]``.
    p_ref : torch.Tensor
        Reference solution with shape ``[C]``.

    Returns
    -------
    dict[str, float]
        Absolute L2, relative L2, and L-infinity error norms.
    """
    diff = p_num - p_ref
    l2_abs = torch.linalg.vector_norm(diff).item()
    l2_ref = torch.linalg.vector_norm(p_ref).item()
    rel_l2 = l2_abs / l2_ref if l2_ref > 0.0 else l2_abs
    return {
        "l2_abs": l2_abs,
        "rel_l2": rel_l2,
        "linf": diff.abs().max().item(),
    }


def characteristic_cell_size(grid: IGridBase) -> float:
    """Return cell size as characteristic mesh spacing."""
    return grid.cell_sizes[0, 0].item()


def solve_poisson_case(
    n_leaf_refinement: int,
    *,
    output_dir: Path,
    save_vtu: bool,
) -> PoissonMetrics:
    """
    Solve the manufactured Poisson problem for one mesh resolution.

    Parameters
    ----------
    n_leaf_refinement : int
        Uniform leaf-refinement level passed to ``config.j2``.
    output_dir : Path
        Directory for optional VTU output.
    save_vtu : bool
        Whether to export VTU files for each case.

    Returns
    -------
    PoissonMetrics
        Cell count and L-infinity error against the analytical solution.
    """
    config = load_config(n_leaf_refinement)
    grid = create_grid_from_config(config)

    p_name = make_field_name("p")
    p = get_or_create_cellfield(grid, p_name, FieldRole.LOCAL, ())
    p_solver = create_solver(grid.sim_config.fvSolution.solvers[p_name])

    cell_centers = grid.cell_centers
    x = cell_centers[:, 0]
    y = cell_centers[:, 1]
    z = cell_centers[:, 2]

    poisson_mat = assemble_poisson_matrix(p)
    poisson_eq = equation(p, poisson_mat)
    p.data = p_solver.solve(poisson_eq).solution

    p_ref = p_exact(x, y, z)
    errors = compute_errors(p.data, p_ref)

    if save_vtu:
        p_exact_name = make_field_name("p_exact")
        diff_name = make_field_name("diff")
        p_exact_field = get_or_create_cellfield(
            grid, p_exact_name, FieldRole.LOCAL, ()
        )
        diff_field = get_or_create_cellfield(
            grid, diff_name, FieldRole.LOCAL, ()
        )
        p_exact_field.export = True
        diff_field.export = True

        p_exact_field.data = p_ref
        diff_field.data = (p.data - p_ref).abs()
        case_output_dir = output_dir / f"n_leaf_{n_leaf_refinement}"
        case_output_dir.mkdir(parents=True, exist_ok=True)
        save_export_fields_as_vtu(
            grid,
            str(case_output_dir / f"poisson_{n_leaf_refinement:02d}.vtu"),
            ugrid=to_unstructured_grid(grid),
        )

    metrics = PoissonMetrics(
        n_leaf_refinement=n_leaf_refinement,
        dx=characteristic_cell_size(grid),
        linf=errors["linf"],
    )
    logger.info(
        "n_leaf_refinement=%d dx=%.4e Linf=%.4e",
        metrics.n_leaf_refinement,
        metrics.dx,
        metrics.linf,
    )

    if grid.device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return metrics


def write_metrics_csv(metrics: list[PoissonMetrics], csv_path: Path) -> None:
    """Write convergence metrics to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["n_leaf_refinement", "dx", "linf"],
        )
        writer.writeheader()
        for row in metrics:
            writer.writerow(row.model_dump())


def plot_linf_vs_dx(metrics: list[PoissonMetrics], plot_path: Path) -> None:
    """Plot L-infinity error against the characteristic cell size."""
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(metrics, key=lambda row: row.dx, reverse=True)
    dx = [row.dx for row in ordered]
    linf = [row.linf for row in ordered]

    dx_ref = ordered[-1].dx
    linf_ref = ordered[-1].linf
    linf_second_order = [linf_ref * (mesh_dx / dx_ref) ** 2 for mesh_dx in dx]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.loglog(dx, linf, marker="o", linewidth=1.5, label="numerical")
    ax.loglog(
        dx,
        linf_second_order,
        linestyle="--",
        linewidth=1.5,
        label="2nd order",
    )
    ax.set_xlabel("dx")
    ax.set_ylabel("L-infinity error")
    ax.set_title("Poisson solver convergence")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve a manufactured Poisson problem while sweeping "
            "n_leaf_refinement and record convergence metrics."
        )
    )
    parser.add_argument(
        "--n-leaf-refinements",
        type=int,
        nargs="+",
        default=list(DEFAULT_N_LEAF_REFINEMENTS),
        help=(
            "Uniform leaf-refinement levels to evaluate. "
            f"Default: {list(DEFAULT_N_LEAF_REFINEMENTS)}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for metrics, plots, and VTU files. "
            f"Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )
    parser.add_argument(
        "--save-vtu",
        action="store_true",
        help="Export VTU files for each refinement level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_run_logger(output_dir / LOG_FILE_NAME)

    metrics: list[PoissonMetrics] = []
    for n_leaf_refinement in args.n_leaf_refinements:
        metrics.append(
            solve_poisson_case(
                n_leaf_refinement,
                output_dir=output_dir,
                save_vtu=args.save_vtu,
            )
        )

    metrics_csv = output_dir / METRICS_CSV_NAME
    convergence_plot = output_dir / CONVERGENCE_PLOT_NAME
    write_metrics_csv(metrics, metrics_csv)
    plot_linf_vs_dx(metrics, convergence_plot)

    logger.info("Wrote metrics to %s", metrics_csv)
    logger.info("Wrote plot to %s", convergence_plot)


if __name__ == "__main__":
    main()
