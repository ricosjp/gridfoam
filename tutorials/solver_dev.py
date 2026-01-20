"""
Compute solver
==========================================
:mod:`gridfoam` can solve basic equations.

This tutorial shows how to compute basic equations.
"""

import pathlib

import torch
from jaxtyping import Float

from gridfoam.CTX.engine import SimulationEngine
from gridfoam.DNA.enum import BoundaryConditionType, FieldLayout, FieldRole
from gridfoam.DNA.meta.boundary_condition import BoundaryConditionMeta
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.RNA.builtins.builtin_fields import builtin_T
from gridfoam.RNA.defaults import default_registry
from gridfoam.RNA.equation.api import ddt, div, equation, laplacian

configpath = pathlib.Path("tests/data/yaml/bunny.yaml")


def initialize_U(
    x: Float[torch.Tensor, "W W W"],
    y: Float[torch.Tensor, "W W W"],
    z: Float[torch.Tensor, "W W W"],
) -> Float[torch.Tensor, "C W W W"]:
    W = x.shape[0]
    device = x.device
    dtype = x.dtype
    U = torch.zeros((3, W, W, W), dtype=dtype, device=device)
    U[0] = 0.1
    return U


def initialize_T(
    x: Float[torch.Tensor, "W W W"],
    y: Float[torch.Tensor, "W W W"],
    z: Float[torch.Tensor, "W W W"],
) -> Float[torch.Tensor, "C W W W"]:
    W = x.shape[0]
    device = x.device
    dtype = x.dtype
    T = torch.zeros((1, W, W, W), dtype=dtype, device=device)
    mask = (
        (-1.0 < x) & (x < 1.0) & (-1.0 < y) & (y < 1.0) & (0.0 < z) & (z < 2.0)
    )
    T[0, mask] = 1.0
    return T


def initialize_nu(
    x: Float[torch.Tensor, "W W W"],
    y: Float[torch.Tensor, "W W W"],
    z: Float[torch.Tensor, "W W W"],
) -> Float[torch.Tensor, "C W W W"]:
    W = x.shape[0]
    device = x.device
    dtype = x.dtype
    nu = torch.full((1, W, W, W), 0.01, dtype=dtype, device=device)
    return nu


if __name__ == "__main__":
    registry = default_registry()
    registry.register_field(builtin_T())
    registry.register_field(
        FieldMeta(
            name="nu",
            label="Thermal diffusivity",
            role=FieldRole.AUXILIARY,
            layout=FieldLayout.CELL,
            components=1,
            unit="m^2/s",
            initialize_func=initialize_nu,
        )
    )

    U = registry.get_field("U")
    T = registry.get_field("T")
    U.initialize_func = initialize_U
    T.initialize_func = initialize_T
    phi = registry.get_field("phi")
    nu = registry.get_field("nu")

    bcs_heat_diffusion = [
        BoundaryConditionMeta(
            name="T_wall",
            target_field=T,
            target_boundary_labels=[
                # "domainZ-",
                # "domainY-",
                "domainX-",
                # "domainX+",
                # "domainY+",
                # "domainZ+",
            ],
            type=BoundaryConditionType.DIRICHLET,
            value=torch.ones((1,), dtype=torch.float64),
        ),
    ]

    eq_heat_diffusion = equation(
        name="heat_diffusion",
        target=T,
        boundary_conditions=bcs_heat_diffusion,
        # lhs=ddt(T) + div(phi, T),
        # lhs=ddt(T) - laplacian(nu, T),
        lhs=ddt(T) + div(phi, T) - laplacian(nu, T),
    )
    registry.register_equation(eq_heat_diffusion)
    simulation_engine = SimulationEngine(
        registry=registry,
        configpath=configpath,
    )
    simulation_engine.initialize()

    # total_T = 0.0
    # for _, cube in simulation_engine.context.grid_handle.iter_all_leaves():
    #     field = cube.field
    #     T_field = field.cells["T"]
    #     cube_T = T_field.interior[0].sum()
    #     total_T += cube_T
    # print(f"Initial total T: {total_T}")

    simulation_engine.solve(eq_heat_diffusion)

    # total_T = 0.0
    # for _, cube in simulation_engine.context.grid_handle.iter_all_leaves():
    #     field = cube.field
    #     cube_T = field.cells["T"].interior[0].sum()
    #     total_T += cube_T
    # print(f"Final total T: {total_T}")
