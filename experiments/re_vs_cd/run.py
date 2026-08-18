import argparse
import csv
import logging
import shutil
import stat
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import mlflow
import torch
import yaml
from jinja2 import Template
from pydantic import BaseModel

from gridfoam.algorithms.factory import create_algorithm
from gridfoam.core.grid.factory import create_grid
from gridfoam.io.vtu import save_export_fields_as_vtu, to_unstructured_grid
from gridfoam.meta.config import GridfoamConfig, ManualAlgorithm
from gridfoam.post.diagnostics import DiagnosticsCollector
from gridfoam.post.forces.coeffs import ForceCoeffs
from gridfoam.post.forces.evaluator import ForceEvaluator

PARAMETERS_PATH = Path("experiments/re_vs_cd/data/parameters.yml")
GRIDFOAM_TEMPLATE_PATH = Path(
    "experiments/re_vs_cd/templates/gridfoam/config.j2"
)
OPENFOAM_TEMPLATE_PATH = Path("experiments/re_vs_cd/templates/openfoam")

MeshLevel = Literal["coarse", "fine"]
SolverName = Literal["gridfoam", "openfoam", "all"]

# OpenFOAM snappyHexMesh / blockMesh presets.
# coarse matches the historical re_vs_cd OpenFOAM setup (~7e5 cells).
# fine raises base resolution and one refinement level for wake checks.
OPENFOAM_MESH_PRESETS: dict[MeshLevel, dict[str, int]] = {
    "coarse": {
        "block_nx": 96,
        "block_ny": 32,
        "block_nz": 32,
        "surface_level": 2,
        "box_level": 2,
        "max_local_cells": 200000,
        "max_global_cells": 2000000,
    },
    "fine": {
        "block_nx": 144,
        "block_ny": 48,
        "block_nz": 48,
        "surface_level": 3,
        "box_level": 3,
        "max_local_cells": 1000000,
        "max_global_cells": 3000000,
    },
}

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


def openfoam_solver_dir_name(mesh_level: MeshLevel = "coarse") -> str:
    return "openfoam" if mesh_level == "coarse" else f"openfoam_{mesh_level}"


def openfoam_output_dir(
    case: CaseConfig, Re: float, mesh_level: MeshLevel = "coarse"
) -> Path:
    return Path(
        "experiments/re_vs_cd/outputs/"
        f"{case.name}/{openfoam_solver_dir_name(mesh_level)}/{re_param(Re)}"
    )


def openfoam_template_context(
    case: CaseConfig, Re: float, mesh_level: MeshLevel = "coarse"
) -> dict[str, str | float | int]:
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
        "mesh_level": mesh_level,
        **OPENFOAM_MESH_PRESETS[mesh_level],
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


def render_openfoam_case(
    case: CaseConfig, Re: float, mesh_level: MeshLevel = "coarse"
) -> Path:
    output_dir = openfoam_output_dir(case, Re, mesh_level=mesh_level)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(OPENFOAM_TEMPLATE_PATH, output_dir)

    context = openfoam_template_context(case, Re, mesh_level=mesh_level)
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


def run_openfoam_case(
    case: CaseConfig, Re: float, mesh_level: MeshLevel = "coarse"
) -> float:
    if shutil.which("blockMesh") is None:
        raise RuntimeError("OpenFOAM not found in PATH")

    case_dir = render_openfoam_case(case, Re, mesh_level=mesh_level)
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

    post_processing = grid.sim_config.post_processing
    assert post_processing is not None
    assert post_processing.forceCoeff is not None
    force_evaluator = ForceEvaluator(grid, phase=phase)

    diagnostics = DiagnosticsCollector.from_config(
        grid,
        post_processing,
        output_dir,
        n_steps=n_steps,
        delta_t=deltaT,
        phase=phase,
    )
    if diagnostics is not None:
        algorithm.attach_diagnostics(diagnostics)

    try:
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
                max_u = (
                    torch.linalg.vector_norm(U.data, ord=2, dim=1).max().item()
                )
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
    finally:
        if diagnostics is not None:
            diagnostics.close()

    return force_evaluator.history[-1].Cd.item()


def parse_args(
    case_names: list[str], re_values: list[float]
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gridfoam / OpenFOAM re_vs_cd cases."
    )
    parser.add_argument(
        "--solver",
        choices=("gridfoam", "openfoam", "all"),
        default="all",
        help="Which solver family to run.",
    )
    parser.add_argument(
        "--cases",
        choices=case_names,
        default=case_names,
        nargs="+",
        help="Cases to run.",
    )
    parser.add_argument(
        "--re",
        type=float,
        nargs="+",
        default=re_values,
        help="Reynolds numbers to run.",
    )
    parser.add_argument(
        "--mesh",
        choices=tuple(OPENFOAM_MESH_PRESETS),
        default="coarse",
        help="OpenFOAM mesh preset (ignored for gridfoam).",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging (useful for local one-off runs).",
    )
    return parser.parse_args()


def _selected_cases(
    parameters: ExperimentParameters, case_names: list[str]
) -> list[CaseConfig]:
    selected = {name for name in case_names}
    return [case for case in parameters.case if case.name in selected]


def run_gridfoam_cases(
    cases: list[CaseConfig], re_values: list[float], *, use_mlflow: bool
) -> None:
    for case in cases:
        for Re in re_values:
            config = load_gridfoam_config(case, Re)
            run_name = f"gridfoam_{case.name}_{Re:g}"
            if use_mlflow:
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

                    Cd = calculate_drag_coefficient(config)
                    mlflow.log_metrics({"Re": Re, "Cd": Cd})
            else:
                print(f"  Run: {run_name}")
                Cd = calculate_drag_coefficient(config)
                print(f"  Cd={Cd}")


def run_openfoam_cases(
    cases: list[CaseConfig],
    re_values: list[float],
    *,
    mesh_level: MeshLevel,
    use_mlflow: bool,
) -> None:
    for case in cases:
        for Re in re_values:
            run_name = (
                f"openfoam_{mesh_level}_{case.name}_{Re:g}"
                if mesh_level != "coarse"
                else f"openfoam_{case.name}_{Re:g}"
            )
            context = openfoam_template_context(case, Re, mesh_level=mesh_level)
            if use_mlflow:
                with mlflow.start_run(nested=True, run_name=run_name):
                    run = mlflow.active_run()
                    assert run is not None
                    rid = run.info.run_id
                    print(f"  Nested run: {run_name} ({rid})")

                    mlflow.log_param("solver", "openfoam")
                    mlflow.log_params(
                        {
                            "case_name": case.name,
                            "mesh_path": case.mesh_path,
                            "Re": Re,
                            "mesh_level": mesh_level,
                            **{
                                key: value
                                for key, value in context.items()
                                if key != "mesh_level"
                            },
                        }
                    )

                    Cd = run_openfoam_case(case, Re, mesh_level=mesh_level)
                    mlflow.log_metrics({"Re": Re, "Cd": Cd})
            else:
                print(f"  Run: {run_name}")
                Cd = run_openfoam_case(case, Re, mesh_level=mesh_level)
                print(f"  Cd={Cd}")


def main() -> None:
    parameters = load_experiment_parameters(PARAMETERS_PATH)
    case_names = [case.name for case in parameters.case]
    args = parse_args(case_names, parameters.Re)
    cases = _selected_cases(parameters, args.cases)
    re_values = args.re
    use_mlflow = not args.no_mlflow

    if use_mlflow:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Low-Re Experiment")

    if args.solver in ("gridfoam", "all"):
        run_gridfoam_cases(cases, re_values, use_mlflow=use_mlflow)

    if args.solver in ("openfoam", "all"):
        run_openfoam_cases(
            cases,
            re_values,
            mesh_level=args.mesh,
            use_mlflow=use_mlflow,
        )


if __name__ == "__main__":
    main()
