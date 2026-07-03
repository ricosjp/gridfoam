import csv
import logging
import shutil
import stat
import subprocess
from collections.abc import Iterator
from pathlib import Path

import mlflow
import torch
import yaml
from jinja2 import Template
from pydantic import BaseModel

from gridfoam.algorithms.factory import create_algorithm
from gridfoam.core.grid.factory import create_grid
from gridfoam.io.vtu import save_export_fields_as_vtu, to_unstructured_grid
from gridfoam.meta.config import GridfoamConfig, ManualAlgorithm
from gridfoam.post.forces.coeffs import ForceCoeffs
from gridfoam.post.forces.evaluator import ForceEvaluator

PARAMETERS_PATH = Path("experiments/re_vs_cd/data/parameters.yml")
GRIDFOAM_TEMPLATE_PATH = Path(
    "experiments/re_vs_cd/templates/gridfoam/config.j2"
)
OPENFOAM_TEMPLATE_PATH = Path("experiments/re_vs_cd/templates/openfoam")

logger = logging.getLogger("gridfoam.experiments.re_vs_cd")


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


def re_param(Re: float) -> str:
    return f"re_{Re:g}"


def compute_nu(case: CaseConfig, Re: float) -> float:
    return case.magU_ref * case.L_ref / Re


def openfoam_output_dir(case: CaseConfig, Re: float) -> Path:
    return Path(
        f"experiments/re_vs_cd/outputs/{case.name}/openfoam/{re_param(Re)}"
    )


def openfoam_template_context(
    case: CaseConfig, Re: float
) -> dict[str, str | float]:
    nu = compute_nu(case, Re)
    mesh_name = Path(case.mesh_path).name
    return {
        "case_name": case.name,
        "mesh_name": mesh_name,
        "re_param": re_param(Re),
        "nu": f"{nu:.16g}",
        "magU_ref": case.magU_ref,
        "A_ref": case.A_ref,
        "L_ref": case.L_ref,
    }


def load_gridfoam_config(case: CaseConfig, Re: float) -> GridfoamConfig:
    template = Template(GRIDFOAM_TEMPLATE_PATH.read_text())
    nu = compute_nu(case, Re)
    rendered = template.render(
        mesh_path=case.mesh_path,
        case_name=case.name,
        re_param=re_param(Re),
        nu=nu,
        magU_ref=case.magU_ref,
        A_ref=case.A_ref,
        L_ref=case.L_ref,
    )
    return GridfoamConfig.model_validate(yaml.safe_load(rendered))


def render_openfoam_case(case: CaseConfig, Re: float) -> Path:
    output_dir = openfoam_output_dir(case, Re)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(OPENFOAM_TEMPLATE_PATH, output_dir)

    context = openfoam_template_context(case, Re)
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text()
        if "{{" in text:
            path.write_text(Template(text).render(**context))

    mesh_src = Path(case.mesh_path)
    tri_surface_dir = output_dir / "constant" / "triSurface"
    tri_surface_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mesh_src, tri_surface_dir / mesh_src.name)

    for script_name in ("Allrun", "Allclean"):
        script_path = output_dir / script_name
        script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)

    return output_dir


def read_openfoam_numeric_rows(path: Path) -> list[list[float]]:
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


def read_openfoam_cd(case_dir: Path) -> float:
    force_coeffs_dir = case_dir / "postProcessing" / "forceCoeffs1" / "0"
    candidates = sorted(force_coeffs_dir.glob("**/coefficient.dat"))
    if not candidates:
        raise RuntimeError("OpenFOAM forceCoeffs output was not found.")
    rows = read_openfoam_numeric_rows(candidates[-1])
    if not rows:
        raise RuntimeError("OpenFOAM forceCoeffs output is empty.")
    return rows[-1][1]


def run_openfoam_case(case: CaseConfig, Re: float) -> float:
    if shutil.which("blockMesh") is None:
        raise RuntimeError("OpenFOAM not found in PATH")

    case_dir = render_openfoam_case(case, Re)
    subprocess.run(["./Allclean"], cwd=case_dir, check=True)
    subprocess.run(["./Allrun"], cwd=case_dir, check=True)
    return read_openfoam_cd(case_dir)


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

    phase = None
    # Create the algorithm
    algorithm_config = grid.sim_config.fvSolution.algorithm
    if isinstance(algorithm_config, ManualAlgorithm):
        raise ValueError("Manual algorithm has no step sequence")
    algorithm = create_algorithm(grid, phase)

    output_dir = Path(grid.sim_config.control.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_run_logger(output_dir / "run.log")

    base_name = grid.sim_config.control.output.base_name
    write_interval = grid.sim_config.control.writeInterval
    end_time = grid.sim_config.control.endTime
    deltaT = grid.sim_config.control.deltaT
    n_steps = int(end_time / deltaT)

    ugrid = to_unstructured_grid(grid)

    n_steps = int(
        grid.sim_config.control.endTime / grid.sim_config.control.deltaT
    )

    post_processing = grid.sim_config.post_processing
    assert post_processing is not None
    assert post_processing.forceCoeff is not None
    force_evaluator = ForceEvaluator(grid, phase=phase)

    for step in range(1, n_steps + 1):
        algorithm.step()
        if step % write_interval == 0 or step == n_steps:
            save_export_fields_as_vtu(
                grid,
                str(output_dir / f"{base_name}_{step:04d}.vtu"),
                ugrid=ugrid,
            )
            U = grid.get_cellfield("U")
            assert U is not None
            max_u = torch.linalg.vector_norm(U.data, ord=2, dim=1).max().item()
            force_evaluator.evaluate(
                grid,
                time=float(step),
                turbulence=algorithm.turbulence,
            )
            cd = force_evaluator.history[-1].Cd.item()
            logger.info(
                "step=%4d max|U|=%.4e Cd=%.6e",
                step,
                max_u,
                cd,
            )

    force_evaluator.write_csv(output_dir / "coefficients.csv")
    force_evaluator.save_surface_mesh(output_dir / "surface_mesh.vtu")

    return force_evaluator.history[-1].Cd.item()


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    # group experiments
    mlflow.set_experiment("Low-Re Experiment")

    parameters = load_experiment_parameters(PARAMETERS_PATH)

    # gridfoam experiments
    for case in parameters.case:
        for Re in parameters.Re:
            config = load_gridfoam_config(case, Re)
            run_name = f"gridfoam_{case.name}_{Re:g}"
            with mlflow.start_run(nested=True, run_name=run_name):
                run = mlflow.active_run()
                assert run is not None
                rid = run.info.run_id
                print(f"  Nested run: {run_name} ({rid})")

                mlflow.log_param("solver", "gridfoam")
                mlflow.log_params(
                    {
                        "case_name": case.name,
                        "mesh_path": case.mesh_path,
                        "Re": Re,
                        **config.model_dump(),
                    }
                )

                # calculate drag coefficient and log to MLflow
                Cd = calculate_drag_coefficient(config)
                metrics = {
                    "Re": Re,
                    "Cd": Cd,
                }
                mlflow.log_metrics(metrics)

    # openfoam experiments
    for case in parameters.case:
        for Re in parameters.Re:
            run_name = f"openfoam_{case.name}_{Re:g}"
            with mlflow.start_run(nested=True, run_name=run_name):
                run = mlflow.active_run()
                assert run is not None
                rid = run.info.run_id
                print(f"  Nested run: {run_name} ({rid})")

                context = openfoam_template_context(case, Re)
                mlflow.log_param("solver", "openfoam")
                mlflow.log_params(
                    {
                        "case_name": case.name,
                        "mesh_path": case.mesh_path,
                        "Re": Re,
                        **context,
                    }
                )

                Cd = run_openfoam_case(case, Re)
                metrics = {
                    "Re": Re,
                    "Cd": Cd,
                }
                mlflow.log_metrics(metrics)


if __name__ == "__main__":
    main()
