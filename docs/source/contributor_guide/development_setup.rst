.. _contributor-development-setup:

Development setup
=================

Prerequisites
-------------

gridfoam requires:

* Python 3.12
* `uv <https://docs.astral.sh/uv/>`_
* a Rust toolchain and C/C++ build tools for the local ``fluxel`` dependency
* Git submodules, including ``lib/fluxel``

Clone the repository with its submodules, or initialize them in an existing
checkout:

.. code-block:: console

   $ git submodule update --init --recursive

Install dependencies
--------------------

Use ``uv`` for all Python dependency management. For a CPU development
environment:

.. code-block:: console

   $ uv sync --extra cpu --group dev

For CUDA, replace ``cpu`` with the supported extra matching the local runtime,
currently ``cu118`` or ``cu124``. The ``cpu`` and CUDA extras conflict and must
not be selected together.

The Makefile's ``dev-install`` target selects ``cu124`` by default:

.. code-block:: console

   $ make dev-install

Use the explicit ``uv sync`` command above on CPU machines or when another CUDA
version is required.

Run checks
----------

Before submitting a change, run:

.. code-block:: console

   $ make lint
   $ make cpu-test
   $ make document

``make lint`` runs basedpyright, Ruff checks, and a Ruff formatting diff.
``make cpu-test`` runs the default test selection with coverage. See
:doc:`testing` for narrower test commands and test placement.

Build documentation
-------------------

Build the HTML documentation with:

.. code-block:: console

   $ make document

The result is written to ``docs/build/``. The Make target removes previously
generated documentation before rebuilding. If gallery rendering is enabled in
the future on a headless machine, run the target through ``xvfb-run``.

Runtime type checks
-------------------

Runtime type checking is enabled by default. Performance-oriented benchmarks
and profiles may disable it:

.. code-block:: console

   $ GRIDFOAM_RUNTIME_TYPE_CHECKS=0 uv run python path/to/script.py

Do not disable runtime checks in ordinary tests merely to hide a shape or type
error.
