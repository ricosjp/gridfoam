import pathlib

import torch
from jaxtyping import Float

from gridfoam.CTX.engine import SimulationEngine
from gridfoam.DNA.enum import BoundaryConditionType, FieldLayout, FieldRole
from gridfoam.DNA.meta.boundary_condition import BoundaryConditionMeta
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.RNA.builtins.builtin_fields import builtin_T
from gridfoam.RNA.defaults import default_registry
from gridfoam.RNA.equation.api import ddt, div, equation

configpath = pathlib.Path("tests/data/yaml/bunny.yaml")


def initialize_U(
    x: Float[torch.Tensor, "N N N"],
    y: Float[torch.Tensor, "N N N"],
    z: Float[torch.Tensor, "N N N"],
) -> Float[torch.Tensor, "C N N N"]:
    N = x.shape[0]
    device = x.device
    dtype = x.dtype
    U = torch.zeros((3, N, N, N), dtype=dtype, device=device)
    U[0] = 0.1
    return U


def initialize_T(
    x: Float[torch.Tensor, "N N N"],
    y: Float[torch.Tensor, "N N N"],
    z: Float[torch.Tensor, "N N N"],
) -> Float[torch.Tensor, "C N N N"]:
    N = x.shape[0]
    device = x.device
    dtype = x.dtype
    T = torch.zeros((1, N, N, N), dtype=dtype, device=device)
    mask = (-1.0 < x) & (x < 1.0) & (-1.0 < y) & (y < 1.0) & (0.0 < z) & (z < 2.0)
    T[0, mask] = 1.0
    return T

def initialize_nu(
    x: Float[torch.Tensor, "N N N"],
    y: Float[torch.Tensor, "N N N"],
    z: Float[torch.Tensor, "N N N"],
) -> Float[torch.Tensor, "C N N N"]:
    N = x.shape[0]
    device = x.device
    dtype = x.dtype
    nu = torch.full((1, N, N, N), 0.01, dtype=dtype, device=device)
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

    bc_heat_diffusion = BoundaryConditionMeta(
        name="T_wall",
        target_field=T,
        target_boundary_labels=[
            "domainX+",
            "domainX-",
            "domainY+",
            "domainY-",
            "domainZ+",
            "domainZ-",
        ],
        type=BoundaryConditionType.DIRICHLET,
        value=1.0,
    )

    eq_heat_diffusion = equation(
        name="heat_diffusion",
        target=T,
        boundary_condition=bc_heat_diffusion,
        lhs=ddt(T) + div(phi, T),
        # lhs=ddt(T) + div(phi, T),
    )
    registry.register_equation(eq_heat_diffusion)
    simulation_engine = SimulationEngine(
        registry=registry,
        configpath=configpath,
    )
    simulation_engine.initialize()
    simulation_engine.solve(eq_heat_diffusion)

    simulation_engine.context.save("bunny.vtkhdf")
