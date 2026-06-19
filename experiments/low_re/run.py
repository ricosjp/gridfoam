import csv
import logging
from collections.abc import Iterator
from pathlib import Path

import mlflow
import torch
import yaml
from jinja2 import Template
from pydantic import BaseModel

from gridfoam.algorithms.simple import SIMPLE
from gridfoam.boundaries.factory import apply_boundary_condition_configs
from gridfoam.core.field import CellField, FieldRole
from gridfoam.core.grid.factory import create_grid
from gridfoam.initialization import initialize_from_dirichlet_patch
from gridfoam.io.vtu import save_export_fields_as_vtu, to_unstructured_grid
from gridfoam.meta.config import GridfoamConfig
from gridfoam.meta.enums import DomainBoundaryPatch
from gridfoam.models.turbulence.laminar import Laminar
from gridfoam.post.forces import ForceCoeffs, ForceEvaluator

PARAMETERS_PATH = Path("experiments/low_re/data/parameters.yml")
GRIDFOAM_TEMPLATE_PATH = Path("experiments/low_re/templates/gridfoam/config.j2")

logger = logging.getLogger("gridfoam.experiments.low_re")

def configure_run_logger(log_file: Path) -> None:
    formatter = logging.Formatter("%(message)s")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

class CaseConfig(BaseModel):
    name: str
    mesh_path: str
    magU_ref: float
    A_ref: float
    L_ref: float

class ExperimentParameters(BaseModel):
    case: list[CaseConfig]
    Re: list[float]

def load_experiment_parameters(yaml_path: Path) -> ExperimentParameters:
    with yaml_path.open() as f:
        return ExperimentParameters.model_validate(yaml.safe_load(f))

def iter_parameter_combinations(
    parameters: ExperimentParameters,
) -> Iterator[tuple[CaseConfig, float]]:
    for case in parameters.case:
        for Re in parameters.Re:
            yield case, Re

def load_gridfoam_config(case: CaseConfig, Re: float) -> GridfoamConfig:
    template = Template(GRIDFOAM_TEMPLATE_PATH.read_text())
    nu = case.magU_ref * case.L_ref / Re
    rendered = template.render(
        mesh_path=case.mesh_path,
        case_name=case.name,
        re_param=f"re_{Re:g}",
        nu=nu,
        magU_ref=case.magU_ref,
        A_ref=case.A_ref,
        L_ref=case.L_ref,
    )
    return GridfoamConfig.model_validate(yaml.safe_load(rendered))

def write_force_coeffs(path: Path, coeffs: list[ForceCoeffs]) -> None:
    fieldnames = [
        "time",
        "Cd",
        "Cs",
        "Cl",
        "CmRoll",
        "CmPitch",
        "CmYaw",
        "Cd_f",
        "Cd_r",
        "Cs_f",
        "Cs_r",
        "Cl_f",
        "Cl_r",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for co in coeffs:
            writer.writerow(
                {
                    "time": co.time,
                    "Cd": co.Cd.item(),
                    "Cs": co.Cs.item(),
                    "Cl": co.Cl.item(),
                    "CmRoll": co.CmRoll.item(),
                    "CmPitch": co.CmPitch.item(),
                    "CmYaw": co.CmYaw.item(),
                    "Cd_f": co.Cd_f.item(),
                    "Cd_r": co.Cd_r.item(),
                    "Cs_f": co.Cs_f.item(),
                    "Cs_r": co.Cs_r.item(),
                    "Cl_f": co.Cl_f.item(),
                    "Cl_r": co.Cl_r.item(),
                }
            )

def calculate_drag_coefficient(config: GridfoamConfig) -> float:
    grid = create_grid(config)
    U = CellField(grid, "U", role=FieldRole.LOCAL, num_components=3)
    p = CellField(grid, "p", role=FieldRole.LOCAL, num_components=1)
    boundary_conditions = grid.sim_config.boundaryConditions
    if boundary_conditions is None:
        raise ValueError("boundaryConditions is required")
    apply_boundary_condition_configs(U, boundary_conditions["U"])
    apply_boundary_condition_configs(p, boundary_conditions["p"])
    initialize_from_dirichlet_patch(
        U, boundary_conditions["U"], DomainBoundaryPatch.X_MINUS
    )
    nu = grid.sim_config.properties.nu
    turbulence = Laminar(grid=grid, nu=nu)

    force_config = grid.sim_config.forceCoeff
    if force_config is None:
        raise ValueError("forceCoeff is required for this example.")
    force_evaluator = ForceEvaluator(force_config)

    algo = SIMPLE(grid=grid, U=U, p=p, turbulence=turbulence)

    output_dir = Path(grid.sim_config.control.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_run_logger(output_dir / "run.log")

    write_interval = grid.sim_config.control.writeInterval
    base_name = grid.sim_config.control.output.base_name
    ugrid = to_unstructured_grid(grid)

    n_steps = int(
        grid.sim_config.control.endTime / grid.sim_config.control.deltaT
    )

    for step in range(1, n_steps + 1):
        algo.step()
        if step % write_interval == 0 or step == n_steps:
            force_evaluator.evaluate(
                grid,
                time=float(step),
                p=p,
                U=U,
                turbulence=turbulence,
            )
            save_export_fields_as_vtu(
                grid,
                str(output_dir / f"{base_name}{step:04d}.vtu"),
                ugrid=ugrid,
            )
            max_u = torch.linalg.vector_norm(U.data, ord=2, dim=1).max().item()
            cd = force_evaluator.history[-1].Cd.item()
            logger.info(
                "step=%4d max|U|=%.4e Cd=%.6e",
                step,
                max_u,
                cd,
            )

    write_force_coeffs(output_dir / "force_coeffs.csv", force_evaluator.history)
    grid.surface_mesh.save(
        output_dir / "surface_mesh.vtu",
        overwrite_features=True,
        overwrite_file=True,
    )
    return force_evaluator.history[-1].Cd.item()


def main() -> None:
    # set MLflow tracking URI (default is ./mlruns)
    mlflow.set_tracking_uri("file:./mlruns")
    # group experiments
    mlflow.set_experiment("Low-Re Experiment")

    parameters = load_experiment_parameters(PARAMETERS_PATH)


    # gridfoam experiments
    for case in parameters.case:
        for Re in parameters.Re:
            config = load_gridfoam_config(case, Re)
            run_name = f"{case.name}_{Re}"
            with mlflow.start_run(nested=True, run_name=run_name):
                run = mlflow.active_run()
                assert run is not None
                rid = run.info.run_id
                print(f"  Nested run: {run_name} ({rid})")

                # save parameters to MLflow
                mlflow.log_params(config.model_dump())

                # calculate drag coefficient and log to MLflow
                Cd = calculate_drag_coefficient(config)
                metrics = {
                    "Re": Re,
                    "Cd": Cd,
                }
                mlflow.log_metrics(metrics)

if __name__ == "__main__":
    main()
