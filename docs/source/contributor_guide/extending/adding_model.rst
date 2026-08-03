.. _contributor-adding-model:

Adding a physical model
=======================

Transport models
----------------

A transport model subclasses ``models.transport.base.TransportModel`` and
provides ``nu()``. Add model-specific configuration as a frozen Pydantic model
with a literal discriminator, then include it in the ``TransportModel``
discriminated union.

Registration requires:

#. A ``TransportModelType`` enum value.
#. A configuration model and union entry in ``meta/config.py``.
#. The implementation in ``models/transport``.
#. A branch in ``models/transport/factory.py``.
#. Parsing, factory, and property-value tests.

``NewtonianTransport`` is the reference implementation.

Turbulence models
-----------------

A turbulence model subclasses ``models.turbulence.base.TurbulenceModel`` and
implements:

``correct(U, phi)``
   Solve or update the closure model after the flow fields change.

``nu_eff()``
   Return effective viscosity with shape ``[C, 1]``.

The base class creates the configured transport model and registers a
cell-centred ``nu_t`` field. Use ``get_or_create_cellfield`` and
``core.name.make_field_name`` for additional model fields so that registry and
phase naming behavior remain consistent.

Registration requires:

#. A ``TurbulenceType`` enum value.
#. A discriminated configuration model and union entry.
#. The implementation in ``models/turbulence``.
#. A branch in ``models/turbulence/factory.py``.
#. Algorithm wiring if the model needs additional solve statistics or
   correction sequencing.
#. Unit and coupled-flow tests.

An implementation file alone does not make a model user-selectable. For
example, the current factory accepts only the laminar model even though other
work-in-progress modules may exist.

Numerical and differentiation rules
-----------------------------------

* Allocate fields and constants with the grid's dtype and device.
* Keep model state in registered fields when other components must look it up.
* Avoid NumPy conversion or tensor detachment in constitutive calculations.
* Document whether ``correct`` mutates fields and when algorithms must call it.
* Test limiting cases, such as zero turbulent viscosity recovering the
  transport-model value.
* Add gradient tests when a model parameter is intended to be optimized.
