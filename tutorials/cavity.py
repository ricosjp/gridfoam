import pathlib

import torch
from jaxtyping import Float

from gridfoam.CTX.application.piso import PISOEngine
from gridfoam.DNA.enum import BoundaryConditionType, FieldLayout, FieldRole
from gridfoam.DNA.meta.boundary_condition import BoundaryConditionMeta
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.RNA.builtins.builtin_fields import builtin_rAU
from gridfoam.RNA.defaults import default_registry
from gridfoam.RNA.equation.api import ddt, div, equation, laplacian

configpath = pathlib.Path("tests/data/yaml/cavity.yaml")


def initialize_U(
    x: Float[torch.Tensor, "W W W"],
    y: Float[torch.Tensor, "W W W"],
    z: Float[torch.Tensor, "W W W"],
) -> Float[torch.Tensor, "C W W W"]:
    W = x.shape[0]
    device = x.device
    dtype = x.dtype
    U = torch.zeros((3, W, W, W), dtype=dtype, device=device)
    mask = y > 0.1
    U[0, mask] = 1.0
    return U


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

def initialize_p(
    x: Float[torch.Tensor, "W W W"],
    y: Float[torch.Tensor, "W W W"],
    z: Float[torch.Tensor, "W W W"],
) -> Float[torch.Tensor, "C W W W"]:
    W = x.shape[0]
    device = x.device
    dtype = x.dtype
    p = torch.zeros((1, W, W, W), dtype=dtype, device=device)
    return p

if __name__ == "__main__":
    registry = default_registry()
    registry.register_field(builtin_rAU())
    registry.register_field(
        FieldMeta(
            name="nu",
            label="Velocity diffusivity",
            role=FieldRole.AUXILIARY,
            layout=FieldLayout.CELL,
            components=1,
            unit="m^2/s",
            initialize_func=initialize_nu,
        )
    )
    registry.register_field(
        FieldMeta(
            name="gradp",
            label="Pressure gradient",
            role=FieldRole.AUXILIARY,
            layout=FieldLayout.CELL,
            components=3,
        )
    )

    U = registry.get_field("U")
    p = registry.get_field("p")
    U.initialize_func = initialize_U
    p.initialize_func = initialize_p
    phi = registry.get_field("phi")
    nu = registry.get_field("nu")
    rAU = registry.get_field("rAU")

    bcs_U = [
        BoundaryConditionMeta(
            name="fixed_wall",
            target_field=U,
            target_boundary_labels=[
                "domainZ-",
                "domainY-",
                "domainX-",
                "domainX+",
                "domainZ+",
            ],
            type=BoundaryConditionType.DIRICHLET,
            value=torch.tensor([0.0, 0.0, 0.0]),
        ),
        BoundaryConditionMeta(
            name="moving_wall",
            target_field=U,
            target_boundary_labels=[
                "domainY+",
            ],
            type=BoundaryConditionType.DIRICHLET,
            value=torch.tensor([1.0, 0.0, 0.0]),
        ),
    ]

    bcs_p = [
        BoundaryConditionMeta(
            name="wall",
            target_field=p,
            target_boundary_labels=[
                "domainZ-",
                "domainY-",
                "domainX-",
                "domainX+",
                "domainY+",
                "domainZ+",
            ],
            type=BoundaryConditionType.NEUMANN,
            value=torch.tensor([0.0, 0.0, 0.0]),
        ),
    ]

    bcs = bcs_U + bcs_p

    momentum_equation = equation(
        name="momentum",
        target=U,
        boundary_conditions=bcs,
        lhs=ddt(U) + div(phi, U) - laplacian(nu, U)
        # lhs=ddt(U) + div(phi, U)
    )
    registry.register_equation(momentum_equation)

    poisson_equation = equation(
        name="poisson",
        target=p,
        boundary_conditions=bcs,
        lhs=laplacian(rAU, p) - div(U)
    )
    registry.register_equation(poisson_equation)

    simulation_engine = PISOEngine(
        registry=registry,
        configpath=configpath,
    )
    simulation_engine.initialize()


    simulation_engine.solve()

#TODO empty boundary
#TODO test
#TODO check convergence
