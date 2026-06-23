from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from gridfoam.algorithms.simple import SIMPLE
from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.boundaries.basic.neumann import NeumannBC
from gridfoam.boundaries.basic.slip import SlipBC
from gridfoam.boundaries.derived.inlet_outlet import InletOutletBC
from gridfoam.core.field import CellField
from gridfoam.core.grid.factory import create_grid
from gridfoam.meta.config import (
    ControlConfig,
    DomainConfig,
    FluxelConfig,
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


@dataclass(frozen=True)
class SimulationResult:
    U: torch.Tensor
    volumes: torch.Tensor
    num_cells: int


def _config(
    *,
    end_time: float,
    output_dir: Path,
    device: DeviceType,
) -> GridfoamConfig:
    return GridfoamConfig(
        fluxel=FluxelConfig(
            domain=DomainConfig(
                lower=[0.0, 0.0, 0.0],
                upper=[1.0, 1.0, 0.1],
            ),
            root_resolution=[4, 4, 1],
            target_level=0,
            n_leaf_refinement=0,
            mesh_path=None,
            ibm_type=IbmType.AXIS_PROJECTED,
        ),
        simulator=SimulatorConfig(
            control=ControlConfig(
                deltaT=1.0,
                endTime=end_time,
                writeInterval=1000,
                output=OutputConfig(
                    output_dir=output_dir,
                    base_name="autograd_velocity_sensitivity",
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
                        tolerance=1e-12,
                        rel_tolerance=0.0,
                        max_iter=16,
                        norm_type=NormType.L_2,
                    ),
                    "pressure_poisson": SolverConfig(
                        method=SolverType.CG,
                        preconditioner=PreconditionerType.JACOBI,
                        tolerance=1e-12,
                        rel_tolerance=0.0,
                        max_iter=16,
                        norm_type=NormType.L_2,
                    ),
                },
                n_non_orthogonal_correctors=0,
            ),
            boundaryConditions=None,
            properties=PropertiesConfig(nu=0.1),
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
        }
    )


def _simulate(
    config: GridfoamConfig,
    *,
    U_init: torch.Tensor,
    U_inlet: torch.Tensor,
    nu: float,
) -> SimulationResult:
    grid = create_grid(config)
    U = CellField(grid, "U", role=FieldRole.LOCAL, num_components=3)
    p = CellField(grid, "p", role=FieldRole.LOCAL, num_components=1)
    U.data = U_init

    _apply_boundary_conditions(U, p, U_inlet=U_inlet)

    algo = SIMPLE(
        grid=grid,
        U=U,
        p=p,
        turbulence=Laminar(grid=grid, nu=nu),
    )
    n_steps = int(
        grid.sim_config.control.endTime / grid.sim_config.control.deltaT
    )
    for _ in range(n_steps):
        algo.step()

    return SimulationResult(
        U=U.data,
        volumes=grid.cell_volumes,
        num_cells=grid.num_cells,
    )


def _weighted_velocity_loss(
    U_sim: torch.Tensor,
    U_ans: torch.Tensor,
    volumes: torch.Tensor,
) -> torch.Tensor:
    diff = U_sim - U_ans
    cell_sq = torch.sum(diff * diff, dim=1, keepdim=True)
    return torch.sqrt(torch.sum(volumes * cell_sq))


def _summarize_grad(name: str, grad: torch.Tensor) -> None:
    grad_norm = torch.linalg.vector_norm(grad).item()
    max_abs = torch.max(torch.abs(grad)).item()
    print(f"{name}: shape={tuple(grad.shape)} ||grad||={grad_norm:.6e}")
    print(f"{name}: max|grad|={max_abs:.6e}")
    if grad.ndim == 2:
        per_component = torch.linalg.vector_norm(grad, dim=0)
        values = ", ".join(f"{value.item():.6e}" for value in per_component)
        print(f"{name}: per-component ||grad||=[{values}]")
    else:
        values = ", ".join(f"{value.item():.6e}" for value in grad.reshape(-1))
        print(f"{name}: values=[{values}]")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Differentiate a volume-weighted velocity mismatch with respect "
            "to U_init and U_inlet."
        )
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Number of SIMPLE steps to unroll. Default: 2.",
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=0.1,
        help="Laminar viscosity. Default: 0.1.",
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
        default=Path("/tmp/gridfoam_autograd_velocity_sensitivity"),
        help="Output directory used by the config. Default: /tmp/...",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device_type = DeviceType(args.device)
    config = _config(
        end_time=float(args.steps),
        output_dir=args.output_dir,
        device=device_type,
    )

    probe_grid = create_grid(config)
    dtype = probe_grid.dtype
    device = probe_grid.device
    num_cells = probe_grid.num_cells

    U_ref_init = torch.zeros((num_cells, 3), dtype=dtype, device=device)
    U_ref_inlet = _constant_vector((1.0, 0.0, 0.0), dtype=dtype, device=device)

    with torch.no_grad():
        ans = _simulate(
            config,
            U_init=U_ref_init.clone(),
            U_inlet=U_ref_inlet.clone(),
            nu=args.nu,
        )
        U_ans = ans.U.detach()

    U_init = torch.zeros(
        (num_cells, 3), dtype=dtype, device=device, requires_grad=True
    )
    U_init.data[:, 0] = 0.15
    sim_from_init = _simulate(
        config,
        U_init=U_init,
        U_inlet=U_ref_inlet.clone(),
        nu=args.nu,
    )
    loss_init = _weighted_velocity_loss(
        sim_from_init.U, U_ans, sim_from_init.volumes
    )
    loss_init.backward()

    U_inlet = _constant_vector((0.75, 0.10, 0.0), dtype=dtype, device=device)
    U_inlet.requires_grad_(True)
    sim_from_inlet = _simulate(
        config,
        U_init=U_ref_init.clone(),
        U_inlet=U_inlet,
        nu=args.nu,
    )
    loss_inlet = _weighted_velocity_loss(
        sim_from_inlet.U, U_ans, sim_from_inlet.volumes
    )
    loss_inlet.backward()

    print(f"num_cells={num_cells} steps={args.steps} device={device}")
    print(f"L(U_init sensitivity)={loss_init.item():.6e}")
    if U_init.grad is None:
        raise RuntimeError("U_init.grad was not populated.")
    _summarize_grad("dL/dU_init", U_init.grad)

    print(f"L(U_inlet sensitivity)={loss_inlet.item():.6e}")
    if U_inlet.grad is None:
        raise RuntimeError("U_inlet.grad was not populated.")
    _summarize_grad("dL/dU_inlet", U_inlet.grad)


if __name__ == "__main__":
    main()
