PISO and PIMPLE physical-time step maps
======================================

``gridfoam.optimize.transient`` provides ``PisoStepMap``, ``PimpleStepMap``
and their shared ``TransientStepMap`` implementation. Each call evaluates
one complete physical time step, returns an independent ``TransientState``,
and restores the caller's fields, histories, iteration controls and diagnostics.
The scope is fixed geometry, laminar flow and first derivatives. Calls sharing
an algorithm must run sequentially.

Step boundary and fixed controls
-------------------------------

PISO includes its momentum predictor, all pressure correctors and all
non-orthogonal passes. PIMPLE additionally includes all outer correctors.
``algorithm.step()`` advances velocity and flux histories once after the
corrections finish. The adapter does not advance them a second time.

``deltaT``, ``nCorrectors``, ``nNonOrthogonalCorrectors`` and PIMPLE's
``nOuterCorrectors`` stay fixed throughout a trajectory. PIMPLE cases must use
``residualControl: {}``; nonempty residual control is rejected rather than
silently changing the case. Changing captured runtime controls or the case
configuration also fails before state application. Internal linear solves
still use their configured convergence criteria. The step contract fixes
pressure/velocity correction counts, not Krylov iteration counts.

DAG inputs and outputs connect only at physical time boundaries. NN corrections
cannot be inserted between internal correctors through these APIs.

State and startup
-----------------

``TransientState.values`` is a ``TensorState`` with these blocks:

* ``U``, ``p``, packed ``phi``, ``rAU``, ``rAtU``, ``HbyA`` and ``phiHbyA``.
* ``U/old`` and ``U/older``: stored velocity time levels.
* ``phi/old`` and ``phi/older``: stored single-sided flux time levels, matching
  ``FaceField.restore_history`` and ``ddtCorr``. These histories do not use the
  packed boundary/immersed-face layout of the current ``phi`` block.

``time``, ``step_index``, ``delta_t`` and ``previous_dt`` are non-differentiable
metadata. ``previous_dt=None`` means the second old level is unavailable;
the older tensor blocks remain present as ignored placeholders so the tensor
layout stays constant across startup. Otherwise ``previous_dt`` must equal
the fixed ``delta_t``. Velocity and flux history validity must agree.
Euler uses the first stored level; backward (BDF2) starts with Euler when
the second level is unavailable. Subsequent steps return valid older levels.

At an unmodified completed-step boundary, ``U/old`` contains the just-completed
velocity and ``U/older`` the preceding velocity. Both stored levels stay fixed
while the next step updates the current predictor and correction fields.

``mapping.initial_state`` is a detached snapshot of the supplied algorithm.
Its default time and index are zero; supply ``initial_time`` and ``initial_step``
when wrapping an already advanced algorithm. An algorithm's diagnostics counter
is not used as its physical clock. To optimize initial conditions, replace the
relevant blocks with parameter-dependent tensors using ``state.with_values``.
When changing the physical initial velocity, also provide its matching first
time level and appropriate flux history.

``state.clone()`` preserves autograd links. ``state.checkpoint()`` deliberately
detaches for snapshot storage and must not be inserted between training steps.
``with_values`` validates layout but does not copy the replacement tensors.
The map clones incoming and outgoing tensors to isolate their storage.

External inputs and execution
----------------------------

As with ``SimpleStepMap``, ``apply_design(algorithm, design)`` is a caller-owned
context manager. It installs every variable boundary/model input and restores
it on exit, including failures. It must not change geometry, the field registry,
configuration or solver settings. Time-dependent inputs are evaluated by the
caller for the intended step and supplied explicitly in ``design``.

The maps prepare and retain scratch fields before capturing their baseline and
freeze field registration during evaluation. Current values are read and
written through ``FieldBindings``; velocity and flux histories are restored
separately. This also lets restoration succeed when an exception traceback
retains an intermediate FVC field. Derived caches are invalidated for replay.
Inference calls evaluate with normal tensors and ``no_grad`` because FV
caches require tensor version counters. SIMPLE and transient maps share this
replay helper.

.. code-block:: python

   from gridfoam.core.state import TensorState
   from gridfoam.optimize.transient import PimpleStepMap

   # algo is a prepared laminar PIMPLE with residualControl={}, fixed deltaT
   # and fixed correction counts. apply_design applies/restores named inputs.
   mapping = PimpleStepMap(algo, apply_design)
   state = mapping.initial_state
   for inlet_velocity in inlet_sequence:
       state = mapping.step(
           state, TensorState({"inlet_velocity": inlet_velocity})
       )
   loss = (state.values["U"][:, 0].mean() - target_speed).square()
   loss.backward()

This differentiates the executed finite sequence of steps. It retains the
trajectory's autograd graph. The executable example is
``examples/optimize/inlet_velocity_transient/run.py``.

Linear-solve failures
-----------------------------------

SIMPLE and transient step maps require successful internal linear solves.
``LinearSolveError`` propagates non-converged component statistics or non-finite
solutions from the primal and CG/BiCGSTAB/PyAMG transpose solves. It identifies
the solve stage and field, and includes the failed component's residual and
iteration count when available. This uses each solver's own residual criterion;
it is separate from physical continuity and outer-loop convergence.

``require_converged_solves`` in ``gridfoam.solvers.base`` can also enforce this
policy for standalone linear solves. Each strict implicit solve retains a copy
of its solver's settings for backward, so leaving the context or running another
forward does not disable the check. Custom solvers must call ``_check_result``
on their transpose result to participate. Ordinary algorithm calls retain their
existing report-only behavior unless the strict policy is enabled.
