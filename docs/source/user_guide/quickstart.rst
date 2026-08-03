.. _user-quickstart:

Quickstart
==========

Run the lid-driven cavity example on CPU. Commands assume the repository root
as the working directory.

Prerequisites
-------------

* Python 3.12
* `uv <https://docs.astral.sh/uv/>`_
* Git submodules initialized (``lib/fluxel``)

Install and run
---------------

.. code-block:: console

   $ git submodule update --init --recursive
   $ uv sync --extra cpu --extra graphlow --group dev
   $ uv run examples/cavity/gridfoam/run.py

The script reads ``examples/cavity/gridfoam/data/config.yaml``, advances a PISO
simulation, and writes VTU files under ``examples/cavity/gridfoam/outputs``.

Next steps
----------

* :doc:`configuration` — YAML structure and main configuration blocks
* :doc:`running_cases` — other bundled examples
* :doc:`../example_gallery/index` — case studies with figures
* :doc:`../api_reference/runner` — programmatic runner API
