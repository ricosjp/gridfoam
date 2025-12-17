from __future__ import annotations

import torch

from gridfoam.DNA.config import SolverChoice
from gridfoam.DNA.enum import FieldLayout, FieldRole, NormType, SolverMethod
from gridfoam.DNA.scheme.solver._CG import CG


class TestCG:
    """Test suite for CG class."""

    def test_required_fields(self) -> None:
        """Test required_fields class method."""
        required = CG.required_fields()

        assert len(required) == 1
        assert required[0].name == "_CG.p"
        assert required[0].label == "Search direction"
        assert required[0].role == FieldRole.AUXILIARY
        assert required[0].layout == FieldLayout.CELL
        assert required[0].components == 1
        assert required[0].dtype == torch.float64
        assert required[0].default_output is False

    def test_init_default(self) -> None:
        """Test initialization with default SolverChoice."""
        solver_choice = SolverChoice(method=SolverMethod.CG)

        cg_solver = CG(solver_choice)

        assert cg_solver.preconditioner is None
        assert cg_solver.tolerance == 1e-6
        assert cg_solver.rel_tolerance == 1e-6
        assert cg_solver.max_iter == 1000
        assert cg_solver.norm_type == NormType.L_2

    def test_init_custom(self) -> None:
        """Test initialization with custom SolverChoice."""
        solver_choice = SolverChoice(
            method=SolverMethod.CG,
            preconditioner="diagonal",
            tolerance=1e-8,
            rel_tolerance=1e-9,
            max_iter=500,
            norm_type=NormType.L_inf,
        )

        cg_solver = CG(solver_choice)

        assert cg_solver.preconditioner == "diagonal"
        assert cg_solver.tolerance == 1e-8
        assert cg_solver.rel_tolerance == 1e-9
        assert cg_solver.max_iter == 500
        assert cg_solver.norm_type == NormType.L_inf

    def test_compute_norm_l2(self) -> None:
        """Test _compute_norm with L_2 norm."""
        solver_choice = SolverChoice(
            method=SolverMethod.CG, norm_type=NormType.L_2
        )
        cg_solver = CG(solver_choice)

        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        norm = cg_solver._compute_norm(x)

        expected = torch.norm(x, p=2).item()
        assert abs(norm - expected) < 1e-6

    def test_compute_norm_l_inf(self) -> None:
        """Test _compute_norm with L_inf norm."""
        solver_choice = SolverChoice(
            method=SolverMethod.CG, norm_type=NormType.L_inf
        )
        cg_solver = CG(solver_choice)

        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        norm = cg_solver._compute_norm(x)

        expected = torch.max(torch.abs(x)).item()
        assert abs(norm - expected) < 1e-6

