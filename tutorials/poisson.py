import pathlib

import torch
from jaxtyping import Float

from gridfoam.CTX.engine import SimulationEngine
from gridfoam.DNA.enum import BoundaryConditionType, FieldLayout, FieldRole
from gridfoam.DNA.meta.boundary_condition import BoundaryConditionMeta
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.RNA.builtins.builtin_fields import builtin_T
from gridfoam.RNA.defaults import default_registry
from gridfoam.RNA.equation.api import equation, laplacian

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
    return T

def initialize_nu(
    x: Float[torch.Tensor, "W W W"],
    y: Float[torch.Tensor, "W W W"],
    z: Float[torch.Tensor, "W W W"],
) -> Float[torch.Tensor, "C W W W"]:
    W = x.shape[0]
    device = x.device
    dtype = x.dtype
    nu = torch.full((1, W, W, W), 1.0, dtype=dtype, device=device)
    return nu

def initialize_f(
    x: Float[torch.Tensor, "W W W"],
    y: Float[torch.Tensor, "W W W"],
    z: Float[torch.Tensor, "W W W"],
) -> Float[torch.Tensor, "C W W W"]:
    W = x.shape[0]
    device = x.device
    dtype = x.dtype
    f = torch.zeros((1, W, W, W), dtype=dtype, device=device)
    def base_func(t):
        return torch.sin(0.25*torch.pi*(t+2))
    f[0] = (3.0*(torch.pi**2)/16.0)*base_func(x) * base_func(y) * base_func(z)
    return f

def initialize_exact_T(
    x: Float[torch.Tensor, "W W W"],
    y: Float[torch.Tensor, "W W W"],
    z: Float[torch.Tensor, "W W W"],
) -> Float[torch.Tensor, "C W W W"]:
    W = x.shape[0]
    device = x.device
    dtype = x.dtype
    exact_T = torch.zeros((1, W, W, W), dtype=dtype, device=device)
    def base_func(t):
        return torch.sin(0.25*torch.pi*(t+2))
    exact_T[0] = base_func(x) * base_func(y) * base_func(z)
    return exact_T

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
    registry.register_field(
        FieldMeta(
            name="f",
            label="Source term",
            role=FieldRole.AUXILIARY,
            layout=FieldLayout.CELL,
            components=1,
            initialize_func=initialize_f,
        )
    )

    registry.register_field(
        FieldMeta(
            name="exact_T",
            label="Exact solution",
            role=FieldRole.AUXILIARY,
            layout=FieldLayout.CELL,
            components=1,
            initialize_func=initialize_exact_T,
        )
    )

    U = registry.get_field("U")
    T = registry.get_field("T")
    U.initialize_func = initialize_U
    T.initialize_func = initialize_T
    phi = registry.get_field("phi")
    nu = registry.get_field("nu")
    f = registry.get_field("f")

    bcs_heat_diffusion = [
        BoundaryConditionMeta(
            name="T_wall",
            target_field=T,
            target_boundary_labels=[
                "domainZ-",
                "domainY-",
                "domainX-",
                "domainX+",
                "domainY+",
                "domainZ+",
            ],
            type=BoundaryConditionType.DIRICHLET,
            value=torch.zeros((1,), dtype=torch.float64),
        ),
    ]

    eq_heat_diffusion = equation(
        name="heat_diffusion",
        target=T,
        boundary_conditions=bcs_heat_diffusion,
        lhs=laplacian(nu, T) + f
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
