import logging

import torch

from gridfoam.algorithms.utils import needs_reference_value, set_reference_value
from gridfoam.boundaries.base import BoundaryCondition
from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.boundaries.basic.neumann import NeumannBC
from gridfoam.core.equation import equation
from gridfoam.core.field import (
    CellField,
    get_or_create_cellfield,
    get_or_create_facefield,
)
from gridfoam.core.grid.base import IGridBase
from gridfoam.core.name import make_field_name
from gridfoam.fv import fvc, fvm
from gridfoam.fv.adjust_phi import adjust_phi
from gridfoam.fv.flux import (
    correct_flux,
    reconstruct_U_from_phi,
)
from gridfoam.meta.config import PotentialFlowConfig, SolverConfig
from gridfoam.meta.enums import BoundaryConditionType, FieldRole
from gridfoam.meta.types import PatchName
from gridfoam.solvers.base import LinearSolver
from gridfoam.solvers.factory import create_solver
from gridfoam.solvers.resolver import resolve_solver_config

logger = logging.getLogger(__name__)


class PotentialFlow:
    """
    OpenFOAM ``potentialFoam``-style velocity-potential initialization.

    Parameters
    ----------
    grid : IGridBase
        Computational grid.
    phase : str | None, optional
        Phase name. Default is None.
    """

    def __init__(
        self,
        grid: IGridBase,
        phase: str | None = None,
    ):
        potential_flow_config = grid.sim_config.fvSolution.potentialFlow
        if potential_flow_config is None:
            raise ValueError(
                "potentialFlow configuration in fvSolution is required."
            )

        U_name = make_field_name("U", phase=phase)
        p_name = make_field_name("p", phase=phase)
        phi_name = make_field_name("phi", phase=phase)

        self.U = get_or_create_cellfield(grid, U_name, FieldRole.LOCAL, 3)
        self.p = get_or_create_cellfield(grid, p_name, FieldRole.LOCAL, 1)
        self.phi = get_or_create_facefield(grid, phi_name, FieldRole.LOCAL, 1)

        boundary_conditions = grid.sim_config.conditions
        if boundary_conditions is None:
            raise ValueError(
                "boundaryConditions is required for potential flow."
            )
        p_condition = boundary_conditions.get(self.p.name)
        if p_condition is None:
            raise ValueError("pressure boundaryConditions are required.")

        self.potential_flow_config = potential_flow_config
        self.solver_configs = grid.sim_config.fvSolution.solvers
        self.p_bc_configs = list(p_condition.iter_bc_configs())
        self.adjust_phi_enabled = grid.sim_config.fvSolution.adjustPhi

    def _set_Phi_bcs(self, Phi: CellField):
        """
        Apply velocity-potential boundary conditions from pressure BC config.

        Dirichlet pressure patches become Dirichlet ``Phi`` patches; all
        other pressure patches become zero-Neumann ``Phi`` patches.

        Parameters
        ----------
        Phi : CellField
            Velocity-potential field receiving boundary conditions.
        """
        bcs: dict[PatchName, BoundaryCondition] = {}
        dtype = self.U.grid.dtype
        device = self.U.grid.device

        for bc_config in self.p_bc_configs:
            if bc_config.type == BoundaryConditionType.DIRICHLET:
                for patch in bc_config.patches:
                    value = torch.tensor(
                        bc_config.value, dtype=dtype, device=device
                    )
                    bcs[patch] = DirichletBC(value)
            else:
                for patch in bc_config.patches:
                    zero_grad = torch.zeros((1,), dtype=dtype, device=device)
                    bcs[patch] = NeumannBC(zero_grad)
        Phi.add_boundary_conditions(bcs)

    def _create_solver(self, config: SolverConfig) -> LinearSolver:
        return create_solver(config)

    def solve(self):
        """
        Solve the velocity-potential equation and reconstruct ``U`` and ``phi``.

        Algorithm (OpenFOAM ``potentialFoam``):

        1. Build ``phi`` from ``U`` and compute ``div(phi)``.
        2. Solve ``laplacian(Phi) = div(phi)`` with non-orthogonal correctors.
        3. Update ``phi -= laplacian(Phi).flux(Phi)`` on internal faces.
        4. Reconstruct ``U`` from ``phi`` and optionally call ``adjustPhi``.
        """
        logger.info("potential flow solve start")

        correct_flux(self.phi, self.U, update_internal=True)
        div_phi = fvc.div(self.phi).data
        continuity_residual = torch.linalg.vector_norm(div_phi, ord=2).item()
        logger.info(
            "potential flow initial continuity residual L2=%.3e",
            continuity_residual,
        )

        Phi = CellField(
            self.U.grid,
            name="Phi",
            role=FieldRole.LOCAL,
            num_components=1,
        )
        Phi.export = False
        self._set_Phi_bcs(Phi)

        phi_solver_config = resolve_solver_config(
            self.solver_configs, "Phi", is_final=False
        )
        potential_solver = self._create_solver(phi_solver_config)
        Phi_needs_ref = needs_reference_value(Phi)

        config = self.potential_flow_config
        assert isinstance(config, PotentialFlowConfig)
        n = config.nNonOrthogonalCorrectors
        phi_ref_cell = config.phiRefCell
        phi_ref_value = (
            config.phiRefValue if config.phiRefValue is not None else 0.0
        )

        phi_eqn_mat = None
        for _ in range(n + 1):
            phi_eqn_mat = fvm.laplacian(1.0, Phi)
            phi_eqn_mat.source = phi_eqn_mat.source + div_phi
            if Phi_needs_ref:
                if phi_ref_cell is None:
                    raise ValueError(
                        "phiRefCell is required when Phi needs a reference "
                        "value"
                    )
                set_reference_value(phi_eqn_mat, phi_ref_cell, phi_ref_value)

            potential_eq = equation(Phi, phi_eqn_mat)
            Phi.data = potential_solver.solve(potential_eq)

        assert phi_eqn_mat is not None
        # phi -= laplacian(Phi).flux(Phi) on single-sided internal faces
        phi_hbya = self.phi.single_data.clone()
        self.phi.single_data = phi_hbya - phi_eqn_mat.flux(Phi.data)

        reconstruct_U_from_phi(self.phi, self.U)
        if self.adjust_phi_enabled:
            adjust_phi(self.phi, self.U)

        continuity_residual = torch.linalg.vector_norm(
            fvc.div(self.phi).data, ord=2
        ).item()
        logger.info(
            "potential flow corrected continuity residual L2=%.3e",
            continuity_residual,
        )
