.. _contributor-testing:

Tests (unit and integration)
============================

Tests live under ``tests/unit/`` and ``tests/integration/``. When unsure where a test belongs, ask whether you need a **real fluxel mesh or multi-component wiring**. If not, prefer unit tests; if yes, use integration tests.

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

From the repository root, CPU tests with coverage::

   make cpu-test

This runs ``uv run pytest tests --cov=src --cov-report term-missing``.
