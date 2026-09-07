.. _contributor-adding-boundary-condition:

Adding a boundary condition
===========================

Boundary conditions convert patch data into the common value-fraction form
used by finite-volume operators.

Contract
--------

Subclass ``boundaries.base.BoundaryCondition`` and implement:

``type``
   The corresponding ``BoundaryConditionType``.

``evaluate(field, patch_name, side)``
   Return ``(fraction, ref_value, ref_grad)`` with shapes ``[F_patch]``,
   ``[F_patch, *component_shape]``, and ``[F_patch, *component_shape]``.

``fraction=1`` represents a Dirichlet contribution and ``fraction=0`` a
Neumann contribution. Mixed conditions may return values between zero and one.

Implementation outline
----------------------

.. code-block:: python

   from gridfoam.boundaries.base import BoundaryCondition


   class ExampleBC(BoundaryCondition):
       @property
       def type(self):
           return BoundaryConditionType.EXAMPLE

       def evaluate(self, field, patch_name, side=FaceSide.UPPER):
           mask = get_mask(field.grid, patch_name, side)
           # Allocate all outputs with field.grid dtype and device.
           ...
           return fraction, ref_value, ref_grad

Use ``boundaries.utils.get_mask`` for domain and immersed patches. Respect the
``side`` argument; it distinguishes upper and lower immersed-boundary values.

Place the implementation in:

* ``boundaries/basic`` when it only uses configured values and local geometry.
* ``boundaries/derived`` when it looks up fields such as ``phi`` or ``rAU``.

Registration checklist
----------------------

#. Add a serialized value to ``BoundaryConditionType`` in ``meta/enums.py``.
#. Extend ``BoundaryConditionConfig`` in ``meta/config.py`` if new parameters
   are required.
#. Add construction logic to ``boundaries/factory.py``.
#. Export the class from the relevant package only if it is part of the public
   package interface.
#. Add YAML parsing and boundary-evaluation tests.

Testing checklist
-----------------

Test:

* output shapes, dtype, and device;
* domain-boundary masks;
* both immersed ``FaceSide`` values when supported;
* missing or invalid configuration values;
* a finite-volume operator using the condition when its assembly behavior is
  new.

Use ``DirichletBC`` or ``NeumannBC`` as the smallest basic examples and
``InletOutletBC`` or ``FixedFluxPressure`` as examples that depend on runtime
field state.
