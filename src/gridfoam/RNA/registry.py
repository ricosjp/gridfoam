from dataclasses import dataclass, field

from gridfoam.DNA.config import fvSchemesConfig, fvSolutionConfig
from gridfoam.DNA.meta.equation import EquationMeta
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.ddt._factory import (
    FVMDdtSchemeConfig,
)
from gridfoam.DNA.scheme.fvm.ddt._interface import IFVMDdtOperator
from gridfoam.DNA.scheme.fvm.div._factory import (
    FVMDivSchemeConfig,
)
from gridfoam.DNA.scheme.fvm.div._interface import IFVMDivOperator
from gridfoam.DNA.scheme.fvm.grad._factory import (
    FVMGradSchemeConfig,
)
from gridfoam.DNA.scheme.fvm.grad._interface import IFVMGradOperator
from gridfoam.DNA.scheme.fvm.laplacian._factory import (
    FVMLaplacianSchemeConfig,
)
from gridfoam.DNA.scheme.fvm.laplacian._interface import IFVMLaplacianOperator
from gridfoam.DNA.scheme.solver._factory import SolverFactory
from gridfoam.DNA.scheme.solver._interface import ILinearSolver


@dataclass(slots=True)
class SimulationMetaRegistry:
    fields: dict[str, FieldMeta] = field(default_factory=dict)
    solvers: dict[str, ILinearSolver] = field(default_factory=dict)
    equations: dict[str, EquationMeta] = field(default_factory=dict)

    ddt_scheme_configs: dict[str, FVMDdtSchemeConfig] = field(
        default_factory=dict
    )
    div_scheme_configs: dict[str, FVMDivSchemeConfig] = field(
        default_factory=dict
    )
    laplacian_scheme_configs: dict[str, FVMLaplacianSchemeConfig] = field(
        default_factory=dict
    )
    grad_scheme_configs: dict[str, FVMGradSchemeConfig] = field(
        default_factory=dict
    )

    # Register field
    def register_field(self, meta: FieldMeta) -> None:
        if meta.name in self.fields:
            msg = f"Field '{meta.name}' is already registered."
            raise ValueError(msg)
        self.fields[meta.name] = meta

    # Register fv schemes
    def register_scheme(self, config: fvSchemesConfig) -> None:
        if config.ddtSchemes is not None:
            for key, choice in config.ddtSchemes.items():
                self.ddt_scheme_configs[key] = FVMDdtSchemeConfig(choice=choice)
        if config.divSchemes is not None:
            for key, choice in config.divSchemes.items():
                self.div_scheme_configs[key] = FVMDivSchemeConfig(choice=choice)
        if config.laplacianSchemes is not None:
            for key, choice in config.laplacianSchemes.items():
                self.laplacian_scheme_configs[key] = FVMLaplacianSchemeConfig(
                    choice=choice
                )
        if config.gradSchemes is not None:
            for key, choice in config.gradSchemes.items():
                self.grad_scheme_configs[key] = FVMGradSchemeConfig(choice=choice)

    # Register linear solver
    def register_solver(self, config: fvSolutionConfig) -> None:
        for eq_name, choice in config.solvers.items():
            if eq_name not in self.equations:
                msg = f"Equation '{eq_name}' is not registered."
                raise ValueError(msg)
            equation_meta = self.equations[eq_name]
            solver_impl = SolverFactory.create(choice, equation_meta)
            self.solvers[eq_name] = solver_impl
            # Register required fields
            for fm in solver_impl.required_fields:
                if fm.name not in self.fields:
                    self.fields[fm.name] = fm

    # Register equation
    def register_equation(self, meta: EquationMeta) -> None:
        if meta.name in self.equations:
            msg = f"Equation '{meta.name}' is already registered."
            raise ValueError(msg)
        self.equations[meta.name] = meta

    # Introspection / query utilities
    def get_field(self, name: str) -> FieldMeta:
        if name not in self.fields:
            msg = f"Field '{name}' is not registered."
            raise ValueError(msg)
        return self.fields[name]

    def get_solver(self, eq_name: str) -> ILinearSolver:
        if eq_name not in self.solvers:
            msg = f"Solver for equation '{eq_name}' is not registered."
            raise ValueError(msg)
        return self.solvers[eq_name]

    def get_equation(self, name: str) -> EquationMeta:
        if name not in self.equations:
            msg = f"Equation '{name}' is not registered."
            raise ValueError(msg)
        return self.equations[name]

    def get_ddt_operator(
        self,
        key: str,
        psi_fm: FieldMeta,
    ) -> IFVMDdtOperator:
        if key not in self.ddt_scheme_configs:
            # Fallback to default
            if "default" in self.ddt_scheme_configs:
                key = "default"
            else:
                msg = f"DDT scheme '{key}' is not registered."
                raise ValueError(msg)
        return self.ddt_scheme_configs[key].create_operator(psi_fm)

    def get_div_operator(
        self, key: str, phi_fm: FieldMeta, psi_fm: FieldMeta
    ) -> IFVMDivOperator:
        if key not in self.div_scheme_configs:
            # Fallback to default
            if "default" in self.div_scheme_configs:
                key = "default"
            else:
                msg = f"DIV scheme '{key}' is not registered."
                raise ValueError(msg)
        return self.div_scheme_configs[key].create_operator(phi_fm, psi_fm)

    def get_laplacian_operator(
        self, key: str, gamma_fm: FieldMeta, psi_fm: FieldMeta
    ) -> IFVMLaplacianOperator:
        if key not in self.laplacian_scheme_configs:
            # Fallback to default
            if "default" in self.laplacian_scheme_configs:
                key = "default"
            else:
                msg = f"LAPLACIAN scheme '{key}' is not registered."
                raise ValueError(msg)
        return self.laplacian_scheme_configs[key].create_operator(
            gamma_fm, psi_fm
        )

    def get_grad_operator(
        self, key: str, psi_fm: FieldMeta
    ) -> IFVMGradOperator:
        if key not in self.grad_scheme_configs:
            # Fallback to default
            if "default" in self.grad_scheme_configs:
                key = "default"
            else:
                msg = f"GRAD scheme '{key}' is not registered."
                raise ValueError(msg)
        return self.grad_scheme_configs[key].create_operator(psi_fm)
