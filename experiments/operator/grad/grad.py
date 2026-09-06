import logging
import sys
from collections.abc import Callable
from pathlib import Path

import pyvista as pv
import torch
import yaml

from gridfoam.core.field import get_or_create_cellfield
from gridfoam.core.grid.base import IGridBase
from gridfoam.core.grid.factory import create_grid as create_grid_from_config
from gridfoam.core.name import make_field_name
from gridfoam.fv import fvc
from gridfoam.io.vtu import save_export_fields_as_vtu, to_unstructured_grid
from gridfoam.meta.config import GridfoamConfig
from gridfoam.meta.enums import FieldRole, GradScheme

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
LOG_FILE_NAME = "grad.log"
RADIAL_EPS = 1.0e-12

logger = logging.getLogger("gridfoam.experiments.operator.grad")

FieldFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
GradFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


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


def create_grid(grad_scheme: GradScheme) -> IGridBase:
    """Build a grid with the requested gradient scheme."""
    with CONFIG_PATH.open() as f:
        raw_yaml = yaml.safe_load(f)
    config = GridfoamConfig.model_validate(raw_yaml)

    fv_schemes = config.simulator.fvSchemes
    updated_fv_schemes = fv_schemes.model_copy(
        update={"gradSchemes": {"default": grad_scheme}}
    )
    simulator = config.simulator.model_copy(
        update={"fvSchemes": updated_fv_schemes}
    )
    config = config.model_copy(update={"simulator": simulator})
    return create_grid_from_config(config)


def p_linear(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return x + y + z


def grad_linear(
    _x: torch.Tensor, _y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    return torch.stack(
        [
            torch.ones_like(z),
            torch.ones_like(z),
            torch.ones_like(z),
        ],
        dim=1,
    )


def p_quadratic_z(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    del x, y
    return (z - 2.0) ** 2


def grad_quadratic_z(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    return torch.stack(
        [
            torch.zeros_like(x),
            torch.zeros_like(y),
            2.0 * (z - 2.0),
        ],
        dim=1,
    )


def p_radial(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(x * x + y * y + z * z)


def grad_radial(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    r = torch.sqrt(x * x + y * y + z * z)
    r = torch.clamp(r, min=RADIAL_EPS)
    return torch.stack([x / r, y / r, z / r], dim=1)


CASES: dict[str, tuple[FieldFn, GradFn]] = {
    "linear": (p_linear, grad_linear),
    "quadratic_z": (p_quadratic_z, grad_quadratic_z),
    "radial": (p_radial, grad_radial),
}

GRAD_SCHEMES = (GradScheme.LINEAR, GradScheme.LEASTSQUARE)


def compute_grad_errors(
    grad_num: torch.Tensor, grad_ref: torch.Tensor
) -> dict[str, float]:
    """
    Compute gradient error norms.

    Parameters
    ----------
    grad_num : torch.Tensor
        Numerical gradient with shape ``[C, 3]``.
    grad_ref : torch.Tensor
        Reference gradient with shape ``[C, 3]``.

    Returns
    -------
    dict[str, float]
        L2 and L-infinity error norms of the gradient vector field.
    """
    diff = grad_num - grad_ref
    return {
        "l2": torch.linalg.vector_norm(diff).item(),
        "linf": diff.abs().max().item(),
    }


def run_case(
    case_name: str,
    p_fn: FieldFn,
    grad_fn: GradFn,
    grad_scheme: GradScheme,
    *,
    output_dir: Path,
    ugrid_cache: dict[str, pv.UnstructuredGrid],
) -> dict[str, float]:
    """
    Evaluate one analytical field and export VTU output.

    Parameters
    ----------
    case_name : str
        Identifier for the manufactured solution.
    p_fn : FieldFn
        Analytical scalar field.
    grad_fn : GradFn
        Analytical gradient field.
    grad_scheme : GradScheme
        Gradient discretization scheme.
    output_dir : Path
        Directory for VTU output.
    ugrid_cache : dict[str, object]
        Cache of unstructured grids keyed by mesh signature.

    Returns
    -------
    dict[str, float]
        Error metrics for the computed gradient.
    """
    grid = create_grid(grad_scheme)
    mesh_key = str(grid.num_cells)
    if mesh_key not in ugrid_cache:
        ugrid_cache[mesh_key] = to_unstructured_grid(grid)
    ugrid = ugrid_cache[mesh_key]

    p_name = make_field_name("p")
    grad_p_name = make_field_name("grad_p")
    grad_exact_name = make_field_name("grad_p_exact")
    grad_err_name = make_field_name("grad_err")

    p = get_or_create_cellfield(grid, p_name, FieldRole.LOCAL, 1)
    grad_p = get_or_create_cellfield(grid, grad_p_name, FieldRole.LOCAL, 3)
    grad_exact = get_or_create_cellfield(
        grid, grad_exact_name, FieldRole.LOCAL, 3
    )
    grad_err = get_or_create_cellfield(grid, grad_err_name, FieldRole.LOCAL, 1)

    grad_p.export = True
    grad_exact.export = True
    grad_err.export = True

    cell_centers = grid.cell_centers
    x = cell_centers[:, 0]
    y = cell_centers[:, 1]
    z = cell_centers[:, 2]

    p.data = p_fn(x, y, z).unsqueeze(-1)
    grad_ref = grad_fn(x, y, z)
    grad_num = fvc.grad(p).data
    grad_p.data = grad_num
    grad_exact.data = grad_ref
    grad_err.data = torch.linalg.vector_norm(
        grad_num - grad_ref, dim=1, keepdim=True
    )

    errors = compute_grad_errors(grad_num, grad_ref)
    output_path = output_dir / f"{case_name}_{grad_scheme.value}.vtu"
    save_export_fields_as_vtu(grid, str(output_path), ugrid=ugrid)

    logger.info(
        "case=%s scheme=%s cells=%d L2=%.4e Linf=%.4e -> %s",
        case_name,
        grad_scheme.value,
        grid.num_cells,
        errors["l2"],
        errors["linf"],
        output_path,
    )
    return errors


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_run_logger(output_dir / LOG_FILE_NAME)

    ugrid_cache: dict[str, pv.UnstructuredGrid] = {}
    for case_name, (p_fn, grad_fn) in CASES.items():
        for grad_scheme in GRAD_SCHEMES:
            run_case(
                case_name,
                p_fn,
                grad_fn,
                grad_scheme,
                output_dir=output_dir,
                ugrid_cache=ugrid_cache,
            )


if __name__ == "__main__":
    main()
