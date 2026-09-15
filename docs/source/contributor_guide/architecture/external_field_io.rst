.. _external-field-io:

External field I/O
==================

Gridfoam exposes ordinary PyTorch APIs for reading and writing prepared
fields and for replaying evaluations on a fixed grid. An external caller
(for example an adapter in another package) owns case loading, any feature
or batch axis, and the lifetime of each case context. Gridfoam does not
import or depend on that package.

Prepare and own fields
----------------------

``create_grid`` constructs the grid, but does not instantiate every field
listed in YAML. The caller prepares its required input, output and scratch
fields using ``CellField``, ``FaceField`` or their ``get_or_create_*`` helpers,
specifying physical component shape, dimensions and temporal role explicitly.

FVC operators lazily create named outputs such as ``grad(psi)`` through
``get_or_create_*``. To use them inside a frozen registry or a checkpointed
replay, prepare these fields beforehand with the same name, shape, dimensions
and role as the operator would use; the operator then reuses the prepared
object. Use distinct field names when distinct storage is required.

``FieldBindings`` holds strong references to existing fields on one grid,
with external keys independent of field names. This keeps bound fields alive
in the grid's weak registry. The caller must also own unbound scratch fields
between evaluations, either directly or through a complete checkpoint.
Replacing a bound field with a new object of the same name invalidates its
bindings; ``read``, ``validate`` and ``write`` then raise ``ValueError``.

``grid.freeze_field_registry()`` pins all currently registered fields and
rejects additions or replacements during its scope. Re-registering the same
object is allowed. Scopes can nest and release the guard after exceptions.
Outside the scope, the existing registration behavior is unchanged. The guard
is enforced by ``GridBase.register_cellfield`` / ``register_facefield``; grid
backends implement only the unguarded ``_register_cellfield`` /
``_register_facefield`` storage methods.

This guard protects membership only. It does not freeze field values,
geometry, boundary conditions or solver settings, and is not a concurrency
lock. Do not run simultaneous evaluations against one grid.

Convert and transfer tensors
----------------------------

Import ``FieldBindings`` from ``gridfoam.core.field_bindings``. It is deliberately
not re-exported through ``core.__init__`` to limit eager import dependencies.

* ``bindings.fields`` exposes a read-only mapping of fields and their metadata.
* ``validate(values)`` checks the key set, physical shapes, dtype, device and
  registered identities before any write. Key order is irrelevant.
* ``write(values)`` clones tensors while preserving gradients, then replaces
  current cell buffers or calls ``FaceField.replace_packed``. No broadcasting,
  conversion or detachment is performed. Field-level FV caches are keyed by
  the data tensor and invalidate themselves; grid-level caches such as face
  geometry are left intact.
* ``read()`` returns a ``TensorState`` with independent storage and live graph
  links. Later field writes do not overwrite these returned values.

Cell tensors have shape ``(C, *physical_shape)``; face tensors have shape
``(packed_n_rows, *physical_shape)``. Face packing follows ``FaceField.pack``:
single-sided faces, domain-boundary faces, then immersed upper and lower
buffers. Use that API's ordering rather than reconstructing boundary
membership externally.

Tensors carry only gridfoam's physical axes. A caller that keeps an
additional feature axis converts ``(C, *physical_shape, F)`` into one gridfoam
evaluation per feature and stacks the results itself. Gridfoam does not
interpret feature metadata; physical dimensions are available from field
metadata and from operator results.

By default ``write`` leaves time histories unchanged. Use
``reset_history=True`` only when initializing a new trajectory; it discards
older levels while retaining gradients through the new initial value.
Use checkpoints to restore an established trajectory. Field bindings are
neither a complete algorithm state nor a boundary-condition binding API.
Segregated step maps in ``gridfoam.optimize`` use the same bindings for the
current ``U``, ``p``, packed ``phi`` and auxiliary fields.

Evaluate and restore
--------------------

The following scalar-gradient example shows one evaluation lifecycle after
creating ``grid``.

.. code-block:: python

   from gridfoam.core.checkpoint import GridCheckpoint
   from gridfoam.core.field import CellField
   from gridfoam.core.field_bindings import FieldBindings
   from gridfoam.fv import fvc
   from gridfoam.meta.enums import FieldRole

   psi = CellField(grid, "psi", FieldRole.LOCAL, ())
   # Let the operator establish its output metadata during preparation.
   gradient = fvc.grad(psi)
   inputs = FieldBindings({"value": psi})
   outputs = FieldBindings({"gradient": gradient})

   def evaluate(value):
       with grid.freeze_field_registry():
           saved = GridCheckpoint.capture(grid, detach=False)
           saved.validate(grid)
           try:
               inputs.write({"value": value})
               fvc.grad(psi)
               return outputs.read()["gradient"]
           finally:
               saved.restore(grid)

For a retained baseline checkpoint, call ``validate`` before each reuse. It
checks grid identity, geometry/topology generations, configuration, time step,
registered field identity and tensor layouts without mutating the case.
Device moves and remeshing require rebuilding the case context and
checkpoints. Checkpoints are tied to live objects and are not a serialization
format for worker processes or training restarts.

Boundary values, runtime solver settings and model parameters remain explicit
replay inputs. Keep them fixed, or reapply and restore them in the caller.
Do not modify geometry while a derivative graph is live; only the explicit
``mark_geometry_changed`` / ``invalidate_derived_caches`` contract is detected.

For evaluation inside ``torch.inference_mode()``, wrap the entire FV execution
and its restore in ``torch.inference_mode(False), torch.no_grad()``. FV caches
need ordinary tensors with version counters; changing only the final output
conversion is insufficient. Construct case contexts outside inference mode.
In training, keep autograd enabled.
