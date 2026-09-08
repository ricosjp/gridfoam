.. _contributor-testing:

Tests
=====

The default test run excludes tests marked ``benchmark``, ``profile``, and
``slow``. When unsure where a test belongs, ask whether it needs a real fluxel
mesh or several production components wired together. If not, prefer a unit
test; if it does, use an integration test.

Writing behavior-focused tests
------------------------------

Describe the contract a failure would break. A module docstring summarizes
the guarantees covered by the file; group them by concern when the file spans
several responsibilities. Test names state the condition and expected behavior,
and a short docstring explains the relevant boundary case or reason for the
check. Keep both descriptions within what the assertions actually verify.
Annotate test functions with ``-> None``.

Keep setup, action, and assertions easy to follow. Use comments for numerical
assumptions, tolerance choices, or non-obvious setup rather than repeating the
docstring. Prefer small helpers that expose the inputs relevant to the test.

Use ``pytest.mark.parametrize`` for independent examples of the same contract,
with descriptive IDs when values alone do not explain the case. Keep sequential
operations together when their order is the behavior under test, such as cache
invalidation, history advancement, or checkpoint replay.

Test a shared rule once where practical, while retaining checks for each
caller's own responsibilities. For example, common scheme lookup precedence
can be exercised through one operator, but every operator still needs a case
where its field-specific key overrides ``default``. Integration tests then
check that the configured scheme changes the numerical operator.

Preserve boundary cases, independent numerical references, and gradient checks
when consolidating tests. For storage isolation, mutate the actual saved-from
tensor and verify that the snapshot is unchanged; mutating an upstream tensor
that already has separate storage cannot establish copying.

Unit tests (``tests/unit/``)
------------------------------

**Goals**

* Validate logic and contracts with minimal dependencies and fast feedback.
* Keep failures easy to localize.

**What belongs here**

* Pure helpers (convergence checks, threshold helpers, etc.).
* Pydantic models and enums.
* Factories and branching (e.g. passing an unknown kind via a mock), without mesh or heavy I/O.

**Rules of thumb**

* **Do not** build fluxel meshes or attach real grids (unless a trivial stub is enough).
* Prefer speed so CI stays cheap.

Integration tests (``tests/integration/``)
------------------------------------------

**Goals**

* Exercise paths where several modules connect, under assumptions closer to production.

**What belongs here**

* Building a mesh via ``create_grid``, then running ``CellField`` / ``FvMatrix`` together with linear solvers.
* Loading example **YAML** from the repo and validating ``GridfoamConfig``.

**Rules of thumb**

* Share a small grid via session-scoped fixtures (e.g. in ``tests/conftest.py``) to amortize mesh cost.
* Keep narrow assertions in unit tests; integration tests should focus on **wiring** correctness.

End-to-end tests (``tests/e2e/``)
----------------------------------

Use end-to-end tests for complete user-visible workflows, including configured
simulation execution and post-processing output. Keep them few and focused;
lower-level numerical behavior belongs in unit or integration tests.

Profile tests (``tests/profile/``)
----------------------------------

Benchmarks and profiling scripts measure runtime or memory behavior. They are
not part of the default correctness suite. Mark expensive pytest cases with the
registered ``benchmark``, ``profile``, or ``slow`` markers as appropriate.

``make benchmark``, ``make profile-time``, and ``make profile-memory`` run CPU
and CUDA sequentially with separate results. Set ``PERF_DEVICES=cpu`` or
``PERF_DEVICES=cuda`` for a single device. To try a short workload:

.. code-block:: console

   $ make profile-time PERF_ARGS="--steps 2 --resolution '5 2 2'"
   $ make profile-memory PERF_DEVICES=cuda PROFILE_ARGS="--native"

Time reports are written to ``tests/profile/outputs/time/{cpu,cuda}/`` and
memory reports to ``tests/profile/outputs/memory/{cpu,cuda}/``. Memory reports
include Memray host allocations and, for CUDA, PyTorch allocator counters and
snapshots. See ``tests/profile/README.md`` for output definitions, benchmark
comparison limits, and the script layout.

Quick comparison
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - Unit (``unit``)
     - Integration (``integration``)
   * - Grid (fluxel)
     - Avoid
     - Use
   * - File I/O (config YAML, etc.)
     - Avoid
     - OK for bundled examples
   * - Main focus
     - Functions, types, configuration
     - Component integration and realistic paths

How to run
----------

Run the complete default CPU suite with coverage:

.. code-block:: console

   $ make cpu-test

Run a focused test while developing:

.. code-block:: console

   $ uv run pytest tests/unit/path/to/test_file.py
   $ uv run pytest tests/unit/path/to/test_file.py::test_name

Run the same static checks used by CI:

.. code-block:: console

   $ make lint

The CPU CI job synchronizes the ``test`` dependency group with the ``cpu`` and
``graphlow`` extras, then runs ``make cpu-test``.

Differentiability tests
-----------------------

When a change affects matrix assembly, a linear solve, or another
differentiable numerical path, test both the primal result and its gradient.
Use double precision and a small deterministic problem for finite-difference
or PyTorch gradient comparisons. Solver changes must preserve the default
implicit-adjoint path; test ``grad_mode="unrolled"`` separately when it is
supported.

Assertions should check tensor shape, dtype, and device when these are part of
the contract. Avoid detaching tensors in the implementation merely to make a
test pass.

Coverage
--------

The full test command is:

.. code-block:: console

   make cpu-test

This runs pytest against ``tests`` with source coverage, missing-line output,
and the five slowest durations.
