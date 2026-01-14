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

configpath = pathlib.Path("tests/data/yaml/SpatialConvergenceTest/poisson.yaml")


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
    def base_func(t: Float[torch.Tensor, "W W W"]) -> Float[torch.Tensor, "W W W"]:
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
    def base_func(t: Float[torch.Tensor, "W W W"]) -> Float[torch.Tensor, "W W W"]:
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

    T = registry.get_field("T")
    T.initialize_func = initialize_T
    nu = registry.get_field("nu")
    f = registry.get_field("f")

    bcs_poisson = [
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

    eq_poisson = equation(
        name="poisson",
        target=T,
        boundary_conditions=bcs_poisson,
        lhs=laplacian(nu, T) + f
    )
    registry.register_equation(eq_poisson)
    simulation_engine = SimulationEngine(
        registry=registry,
        configpath=configpath,
    )
    simulation_engine.initialize()
    simulation_engine.solve(eq_poisson)
