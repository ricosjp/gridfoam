.. _contributor-adding-algorithm:

Adding or changing an algorithm
===============================

Algorithms own the order in which equations are assembled, solved, corrected,
and reported. Reusable discretization belongs in ``fv``; algorithms should
compose those operators rather than reimplement them.

Contract
--------

Subclass ``algorithms.base.AlgorithmBase`` and provide:

``grid``
   The algorithm's ``IGridBase``.

``turbulence``
   Its active ``TurbulenceModel``.

``step()``
   Advance one algorithm step.

At the end of a completed step, call the shared diagnostics finalization with
the corrected face flux and solver statistics. Preserve the expected history
update timing for transient fields.

Shared pressure correction
--------------------------

SIMPLE, PISO, and PIMPLE share pressure-equation and flux-correction helpers
in ``algorithms/utils/pressure_correction.py`` (``solve_pressure_poisson``,
``apply_simplec``, ``correct_phi``, ``correct_velocity``). Put behavior there
when its mathematics and ordering are genuinely common. Algorithm-specific
loop counts, relaxation, solver-key selection, and convergence decisions
remain in the concrete class. SIMPLE always solves pressure with ``p``.
PISO/PIMPLE pass ``final_solver`` as ``pFinal`` only on the last inner
corrector's last non-orthogonal pass.

Assemble explicit sources before constructing ``equation(field, matrix)``.
For momentum prediction, use ``with_source`` to add pressure to a separate
matrix and constrain the pressure-free matrix separately for ``H/A``.
Pass the original pressure matrix to flux correction, since the constrained
solve matrix has eliminated face couplings. Explicit updates must restore
immersed Dirichlet cell values. The details are in
:ref:`architecture-data-contracts`.

Other shared helpers include:

* ``residual.py`` for OpenFOAM-style residual control;
* ``reference_value.py`` for singular pressure-system references.

Registration checklist
----------------------

#. Add an ``AlgorithmType`` value.
#. Add a frozen, discriminator-based configuration model.
#. Include it in the ``Algorithm`` union in ``meta/config.py``.
#. Implement the class under ``algorithms``.
#. Register it in ``algorithms/factory.py``.
#. Confirm how ``runner.manual_step`` should instantiate and step it.
#. Add configuration, step-level, convergence, and end-to-end tests.

Testing checklist
-----------------

Test more than successful execution:

* equation and corrector ordering;
* configured loop counts;
* pressure reference behavior for all-Neumann pressure boundaries;
* final face-flux continuity;
* residual-control stopping;
* field-history updates for transient algorithms;
* solver statistics passed to diagnostics;
* consistency across SIMPLE/PISO/PIMPLE when shared helpers change.

Use small deterministic cases for integration tests. A full example case is
appropriate only when validating the complete user workflow.
