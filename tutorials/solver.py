import pathlib

from gridfoam.CTX.engine import SimulationEngine
from gridfoam.DNA.enum import BoundaryConditionType, FieldLayout, FieldRole
from gridfoam.DNA.meta.boundary_condition import BoundaryConditionMeta
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.RNA.builtins.builtin_fields import builtin_T
from gridfoam.RNA.defaults import default_registry
from gridfoam.RNA.equation.api import ddt, div, equation, laplacian

configpath = pathlib.Path("tests/data/yaml/bunny.yaml")

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
        )
    )

    U = registry.get_field("U")
    T = registry.get_field("T")
    phi = registry.get_field("phi")
    nu = registry.get_field("nu")

    bc_heat_diffusion = BoundaryConditionMeta(
        name="T_wall",
        target_field=T,
        target_boundary_labels=["domainX+", "domainX-", "domainY+", "domainY-", "domainZ+", "domainZ-"],
        type=BoundaryConditionType.DIRICHLET,
        value=293.0,
    )

    eq_heat_diffusion = equation(
        name="heat_diffusion",
        target=T,
        boundary_condition=bc_heat_diffusion,
        lhs=ddt(T) + div(phi, T) - laplacian(nu, T),
    )
    registry.register_equation(eq_heat_diffusion)
    simulation_engine = SimulationEngine(
        registry=registry,
        configpath=configpath,
    )
    simulation_engine.initialize()
    simulation_engine.solve(eq_heat_diffusion)
