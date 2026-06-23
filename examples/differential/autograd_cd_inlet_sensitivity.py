from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from gridfoam.algorithms.simple import SIMPLE
from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.boundaries.basic.neumann import NeumannBC
from gridfoam.boundaries.basic.slip import SlipBC
from gridfoam.boundaries.derived.inlet_outlet import InletOutletBC
from gridfoam.core.field import CellField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.factory import create_grid
from gridfoam.meta.config import (
    ControlConfig,
    DomainConfig,
    FluxelConfig,
    ForceCoeffConfig,
    GridfoamConfig,
    OutputConfig,
    PropertiesConfig,
    SimulatorConfig,
    SolverConfig,
    fvSchemesConfig,
    fvSolutionConfig,
)
from gridfoam.meta.enums import (
    DeviceType,
    DivScheme,
    DomainBoundaryPatch,
    FieldRole,
    IbmType,
    NormType,
    PrecisionType,
    PreconditionerType,
    SolverType,
)
from gridfoam.models.turbulence.laminar import Laminar
from gridfoam.post.forces import ForceEvaluator

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config(
    *,
    steps: int,
    root_resolution: list[int],
    target_level: int,
    output_dir: Path,
    device: DeviceType,
) -> GridfoamConfig:
    return GridfoamConfig(
        fluxel=FluxelConfig(
            domain=DomainConfig(
                lower=[-2.0, -1.5, -1.5],
                upper=[4.0, 1.5, 1.5],
            ),
            root_resolution=root_resolution,
            target_level=target_level,
            n_leaf_refinement=0,
            mesh_path=(
                REPO_ROOT
                / "examples"
                / "low_re_sphere"
                / "gridfoam"
                / "data"
                / "sphere.stl"
            ),
            ibm_type=IbmType.AXIS_PROJECTED,
        ),
        simulator=SimulatorConfig(
            control=ControlConfig(
                deltaT=1.0,
                endTime=float(steps),
                writeInterval=max(steps, 1),
                output=OutputConfig(
                    output_dir=output_dir,
                    base_name="autograd_cd_inlet_sensitivity",
                ),
                precision=PrecisionType.FLOAT64,
            ),
            fvSchemes=fvSchemesConfig(
                divSchemes={"default": DivScheme.UPWIND},
            ),
            fvSolution=fvSolutionConfig(
                solvers={
                    "momentum": SolverConfig(
                        method=SolverType.BiCGSTAB,
                        preconditioner=PreconditionerType.JACOBI,
                        tolerance=1.0e-10,
                        rel_tolerance=0.0,
                        max_iter=128,
                        norm_type=NormType.L_2,
                    ),
                    "pressure_poisson": SolverConfig(
                        method=SolverType.CG,
                        preconditioner=PreconditionerType.JACOBI,
                        tolerance=1.0e-10,
                        rel_tolerance=0.0,
                        max_iter=128,
                        norm_type=NormType.L_2,
                    ),
                },
                n_non_orthogonal_correctors=0,
            ),
            boundaryConditions=None,
            properties=PropertiesConfig(nu=0.1),
            forceCoeff=ForceCoeffConfig.model_validate(
                {
                    "patches": ["_default"],
                    "rho": 1.0,
                    "magU_ref": 1.0,
                    "A_ref": math.pi / 4.0,
                    "L_ref": 1.0,
                    "local_coord": {
                        "drag_dir": [1.0, 0.0, 0.0],
                        "lift_dir": [0.0, 0.0, 1.0],
                        "center_of_rotation": [0.0, 0.0, 0.0],
                    },
                }
            ),
            device=device,
        ),
    )


def _constant_vector(
    values: tuple[float, float, float],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, device=device)


def _constant_scalar(
    value: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor([value], dtype=dtype, device=device)


def _apply_boundary_conditions(
    U: CellField,
    p: CellField,
    *,
    U_inlet: torch.Tensor,
) -> None:
    grid = U.grid
    dtype = grid.dtype
    device = grid.device
    zero_u = _constant_vector((0.0, 0.0, 0.0), dtype=dtype, device=device)
    zero_p = _constant_scalar(0.0, dtype=dtype, device=device)

    U.add_boundary_conditions(
        {
            DomainBoundaryPatch.X_MINUS: DirichletBC(U_inlet),
            DomainBoundaryPatch.X_PLUS: InletOutletBC(zero_u),
            DomainBoundaryPatch.Y_MINUS: SlipBC(),
            DomainBoundaryPatch.Y_PLUS: SlipBC(),
            DomainBoundaryPatch.Z_MINUS: SlipBC(),
            DomainBoundaryPatch.Z_PLUS: SlipBC(),
            "_default": DirichletBC(zero_u),
        }
    )
    p.add_boundary_conditions(
        {
            DomainBoundaryPatch.X_MINUS: NeumannBC(zero_p),
            DomainBoundaryPatch.X_PLUS: DirichletBC(zero_p),
            DomainBoundaryPatch.Y_MINUS: NeumannBC(zero_p),
            DomainBoundaryPatch.Y_PLUS: NeumannBC(zero_p),
            DomainBoundaryPatch.Z_MINUS: NeumannBC(zero_p),
            DomainBoundaryPatch.Z_PLUS: NeumannBC(zero_p),
            "_default": NeumannBC(zero_p),
        }
    )


def _simulate_cd(
    config: GridfoamConfig,
    *,
    U_inlet: torch.Tensor,
    re_value: float,
    couple_initial_field: bool,
) -> tuple[torch.Tensor, AxisProjectedGrid]:
    grid = create_grid(config)
    if not isinstance(grid, AxisProjectedGrid):
        raise TypeError("dCd/dU_inlet example requires an axis-projected grid.")

    U = CellField(grid, "U", role=FieldRole.LOCAL, num_components=3)
    p = CellField(grid, "p", role=FieldRole.LOCAL, num_components=1)

    initial_inlet = U_inlet if couple_initial_field else U_inlet.detach()
    U.data = initial_inlet.reshape(1, 3).expand(grid.num_cells, 3).clone()
    _apply_boundary_conditions(U, p, U_inlet=U_inlet)

    force_config = grid.sim_config.forceCoeff
    if force_config is None:
        raise ValueError("forceCoeff is required for Cd evaluation.")

    nu = force_config.magU_ref * force_config.L_ref / re_value
    turbulence = Laminar(grid=grid, nu=nu)
    algo = SIMPLE(grid=grid, U=U, p=p, turbulence=turbulence)

    n_steps = int(
        grid.sim_config.control.endTime / grid.sim_config.control.deltaT
    )
    for _ in range(n_steps):
        algo.step()

    coeffs = ForceEvaluator(force_config).evaluate(
        grid,
        time=float(n_steps),
        p=p,
        U=U,
        turbulence=turbulence,
    )
    return coeffs.Cd.squeeze(), grid


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Differentiate sphere drag coefficient Cd with respect to the "
            "Dirichlet inlet velocity U_inlet."
        )
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of SIMPLE steps to unroll. Default: 1.",
    )
    parser.add_argument(
        "--re",
        type=float,
        default=1.0,
        help="Reynolds number used to set nu=U_ref L_ref/Re. Default: 1.",
    )
    parser.add_argument(
        "--inlet",
        type=float,
        nargs=3,
        default=[1.0, 0.0, 0.0],
        metavar=("Ux", "Uy", "Uz"),
        help="Inlet velocity vector. Default: 1 0 0.",
    )
    parser.add_argument(
        "--root-resolution",
        type=int,
        nargs=3,
        default=[8, 4, 4],
        metavar=("Nx", "Ny", "Nz"),
        help="Root octree resolution. Default: 8 4 4.",
    )
    parser.add_argument(
        "--target-level",
        type=int,
        default=1,
        help="Surface refinement target level. Default: 1.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Torch device. Default: cpu.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/gridfoam_autograd_cd_inlet_sensitivity"),
        help="Output directory used by the config. Default: /tmp/...",
    )
    parser.add_argument(
        "--couple-initial-field",
        action="store_true",
        help=(
            "Also let the initial cell velocity depend on U_inlet. By default "
            "only the inlet boundary condition is differentiated."
        ),
    )
    parser.add_argument(
        "--detect-anomaly",
        action="store_true",
        help="Enable torch autograd anomaly detection.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be at least 1.")
    if args.re <= 0.0:
        raise ValueError("--re must be positive.")

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    device_type = DeviceType(args.device)
    config = _config(
        steps=args.steps,
        root_resolution=args.root_resolution,
        target_level=args.target_level,
        output_dir=args.output_dir,
        device=device_type,
    )

    device = device_type.to_torch_device()
    U_inlet = torch.tensor(
        args.inlet,
        dtype=PrecisionType.FLOAT64.to_torch_dtype(),
        device=device,
        requires_grad=True,
    )

    Cd, grid = _simulate_cd(
        config,
        U_inlet=U_inlet,
        re_value=args.re,
        couple_initial_field=args.couple_initial_field,
    )
    Cd.backward()

    if U_inlet.grad is None:
        raise RuntimeError(
            "U_inlet.grad is None; Cd is not connected to U_inlet."
        )

    grad_values = ", ".join(f"{value:.6e}" for value in U_inlet.grad.tolist())
    print(f"cells={grid.num_cells} immersed_faces={grid.num_immersed_faces}")
    print(f"Cd={Cd.detach().item():.6e}")
    print(f"dCd/dU_inlet=[{grad_values}]")


if __name__ == "__main__":
    main()
