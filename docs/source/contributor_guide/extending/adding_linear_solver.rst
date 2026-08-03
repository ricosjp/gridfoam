.. _contributor-adding-linear-solver:

Adding a linear solver or preconditioner
========================================

Linear solver contract
----------------------

Subclass ``solvers.base.LinearSolver`` and implement:

``_solve_primal(eq)``
   Solve the equation and return ``SolveResult``. Statistics are reported per
   field component.

``solve_transpose(A_T, rhs)``
   Solve the transposed system for the implicit adjoint.

Do not override ``solve`` only to bypass the base implementation. It controls
the differentiation mode: the default primal solve runs under
``torch.no_grad()``, then attaches an implicit adjoint. Krylov implementations
may support ``grad_mode="unrolled"`` through the same entry point.
``grad_mode`` is currently set on the solver instance; it is not part of
``SolverConfig`` YAML.

``.detach().cpu().numpy()`` is acceptable only when an external library such
as PyAMG requires a host sparse matrix for the primal solve. The public
solution and transpose path must still return PyTorch tensors for the
implicit-adjoint interface.

Registration checklist
----------------------

#. Add a value to ``SolverType`` in ``meta/enums.py``.
#. Extend ``SolverConfig`` only when the solver has new user-facing options.
#. Add the implementation under ``solvers``.
#. Register it in ``solvers/factory.py``.
#. Ensure OpenFOAM-style keys such as ``pFinal`` still resolve through
   ``solvers/resolver.py``.
#. Add convergence, multi-component, transpose, and adjoint tests.

Solver behavior
---------------

Use the shared ``SolveStats`` and ``SolveResult`` types. A solver must apply
``atol``, ``rtol``, and its iteration limit consistently and report whether it
converged. Preserve the input matrix and equation field; callers may reuse
assembled coefficients and inspect solver statistics.

Use helpers in ``solvers/krylov.py`` when adding another Krylov method so that
component-wise and transpose behavior remains consistent.

Adding a preconditioner
-----------------------

Subclass ``solvers.preconditioners.Preconditioner`` and implement its apply
contract. Then:

#. Add a ``PreconditionerType`` value.
#. Register it in ``create_preconditioner``.
#. Add focused tests under ``tests/unit/solvers``.
#. Test it through at least one compatible linear solver.

Document matrix restrictions. For example, a solver or preconditioner that
requires symmetry or positive definiteness must fail clearly or be selected
only for equations that satisfy that requirement.

Differentiation tests
---------------------

Test the primal solution first, then compare gradients of the implicit-adjoint
path against a small dense reference or finite differences. Include gradients
with respect to the LDU coefficients and source when the solver supports them.
For iterative PyTorch solvers, compare the unrolled mode separately.

External-library solvers may convert the primal matrix to another sparse
format, but the public solution must return with the expected PyTorch dtype,
device, and shape, and ``solve_transpose`` must remain available for the
adjoint path.
