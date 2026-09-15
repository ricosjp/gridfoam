Steady solve and SIMPLE replay
========================================

``SimpleStepMap`` and ``steady_solve`` evaluate laminar SIMPLE on a fixed
grid. CPU float64 is the validated configuration.

State and inputs
----------------

``SimpleStepMap`` exposes a named ``TensorState`` containing ``U``, ``p``,
``phi``, ``rAU``, ``rAtU``, ``HbyA`` and ``phiHbyA``. Face tensors use the
existing packed layout, including both APIBM sides. Some auxiliary fields
are read by ``fixedFluxPressure`` before the predictor overwrites them; a
U/p/phi-only vector is not a complete step state for these boundaries.

The full checkpoint additionally contains histories, unrelated registered
fields and iteration controls. Each evaluation restores the baseline, writes
the explicit state and applies the design context.
It suppresses diagnostic files and restores the caller's field/iteration
state even on exceptions. Caches are cleared before replay and on
restoration. Calls sharing an algorithm must run sequentially. SIMPLE and
transient maps share this replay lifecycle.

Design applicators are context managers: install all variable inputs on entry
and restore them on exit. The inlet-velocity example defines a local applicator
that accepts a three-component tensor under the ``inlet_velocity`` key and
replaces the inlet BC only within the context. Applicators can consume named NN
outputs or parameters without changing the tensor API. They must not change
geometry, configuration, field registration or solver settings. Fixed inputs
must remain fixed between forward and backward. Additional evolving model
state needs an explicit adapter; the SIMPLE adapter does not infer arbitrary
hidden state.

Forward and backward
--------------------

``steady_solve`` iterates without retaining the primal trajectory and checks
``||G(q, design)[k] - q[k]|| <= atol + rtol * ||q[k]||`` for every state block.
It returns the state at which that residual was checked. Exhausting the
budget or encountering non-finite values raises ``ConvergenceError``.
This convergence criterion is independent of SIMPLE's diagnostic residual
history. The initial state is an initial guess, not a differentiable trajectory.

Backward receives arbitrary cotangents from the objective and solves
``(I - dG/dq)^T lambda = grad_output`` using matrix-free VJPs and restarted
GMRES. The Hessenberg (Arnoldi least-squares) residual is the primary
iteration check. The true unpreconditioned residual is computed when that
estimate meets the tolerance, at restart or breakdown, and every
``true_residual_interval`` steps; a solve is accepted only after the true
residual meets the tolerance. GMRES avoids requiring the Richardson
iteration itself to be contractive; it still may fail to converge and will
then raise rather than return an unchecked gradient.
The existing implicit linear-solver adjoints supply the VJPs through each
SIMPLE linear solve. Linear-solver tolerances must be tight enough for the
requested gradient accuracy.

The step map requires converged internal primal and transpose solves;
failures raise ``LinearSolveError`` from ``gridfoam.solvers.base``. This policy
persists through backward after the forward scope has restored the caller's
solver settings. See :doc:`transient_steps` for the shared failure contract.

The design gradient is ``(dG/ddesign)^T lambda``. Explicit objective dependence
on design is accumulated by ordinary PyTorch autograd. Each backward builds
its own replay graph. Inference mode is evaluated with normal tensors under
``no_grad`` because FV cache tokens need tensor version counters.

Example
-------

.. code-block:: python

   from gridfoam.core.state import TensorState
   from gridfoam.optimize.simple import SimpleStepMap
   from gridfoam.optimize.steady import steady_solve

   # apply_design is a caller-owned context manager that applies and restores
   # all variable boundary or model inputs.
   mapping = SimpleStepMap(algo, apply_design)
   design = TensorState({"inlet_velocity": inlet_parameter})
   result = steady_solve(mapping, mapping.initial_state, design)
   loss = (result["U"][:, 0].mean() - target_speed).square()
   loss.backward()

The executable example is ``examples/optimize/inlet_velocity_steady/run.py``.
