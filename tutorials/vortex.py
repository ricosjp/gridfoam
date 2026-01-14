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

configpath = pathlib.Path("tests/data/yaml/vortex.yaml")


def initialize_U(
    x: Float[torch.Tensor, "W W W"],
    y: Float[torch.Tensor, "W W W"],
    z: Float[torch.Tensor, "W W W"],
) -> Float[torch.Tensor, "C W W W"]:
    W = x.shape[0]
    device = x.device
    dtype = x.dtype
    U = torch.zeros((3, W, W, W), dtype=dtype, device=device)
    U[0] = -(y - 0.5)
    U[1] = (x - 0.5)
    U[2] = 0.0
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
        (0.25 < x) & (x < 0.75) & (0.25 < y) & (y < 0.75)
        & (0.0 < z) & (z < 0.1)
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
    nu = torch.full((1, W, W, W), 0.001, dtype=dtype, device=device)
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

    bcs_T = [
        BoundaryConditionMeta(
            name="zero_flux",
            target_field=T,
            target_boundary_labels=[
                "domainZ-",
                "domainY-",
                "domainX-",
                "domainX+",
                "domainY+",
                "domainZ+",
            ],
            type=BoundaryConditionType.NEUMANN,
            value=torch.tensor([0.0, 0.0, 0.0]),  # ∂T/∂n = 0 (flux = 0)
        ),
    ]

    eq_heat_diffusion = equation(
        name="heat_diffusion",
        target=T,
        boundary_conditions=bcs_T,
        lhs=ddt(T) + div(phi, T) - laplacian(nu, T),
        # lhs=ddt(T) + div(phi, T),
    )
    registry.register_equation(eq_heat_diffusion)
    simulation_engine = SimulationEngine(
        registry=registry,
        configpath=configpath,
    )
    simulation_engine.initialize()

    simulation_engine.solve(eq_heat_diffusion)
