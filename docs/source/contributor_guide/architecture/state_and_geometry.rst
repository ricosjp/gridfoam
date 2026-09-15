.. _state-and-geometry:

State, replay, and geometry contracts
=====================================

Replayable evaluations keep YAML, public solver APIs, and APIBM thin-sheet
and baffle support. Checkpoints are in-memory objects tied to a live grid.

Package ownership
-----------------

Fluxel owns surface input, grid construction, search, and IBM geometry.
Gridfoam owns numerical state, discretization, physical models, and solve/VJP
APIs. Gridfoam depends on ``phlower-tensor``; it does not depend on phlower.

Differentiable blocks and design inputs
---------------------------------------

``core.state.TensorState`` is an ordered mapping of named tensor blocks. It
supports heterogeneous shapes and preserves the supplied tensor graph, so
boundary inputs, geometry tensors, and neural-network parameters need not be
flattened into one scalar design variable. Integer selections and connectivity
belong in metadata, outside the differentiable blocks.

``as_tuple`` and ``from_tuple`` bridge to autograd VJPs with a strict layout.
Key order, shape, dtype and device must agree; ``validate_layout`` checks this
contract against a named mapping without constructing a state or copying
tensors.

* ``clone()`` copies storage and preserves the graph.
* ``checkpoint()`` copies storage and discards the graph.

Boundary conditions, material/model parameters, and solver settings are explicit
inputs to replay. They are not implicitly captured as evolving field state.
Reapply them for every evaluation, including after restoring a checkpoint.
External adapters can use the same tensor-input contract for boundary
conditions, model parameters and learned corrections.

Field checkpoints
-----------------

``core.checkpoint.GridCheckpoint`` captures all registered cell and face fields,
including an added temperature field, model fields, and scratch fields. Strong
field references keep the grid's weak registry alive. This complete replay
checkpoint is distinct from the conservative unknown vector used by a step map
(``U``, ``p``, packed ``phi`` and the SIMPLE auxiliaries).

Cell checkpoints contain current, old and optional second-old values. Face
checkpoints use the field's packed current values and its single-face flux
history. Each field retains its previous time-step size and any current/old
alias. This preserves the distinction between BDF2 startup and established
two-level history. Restoring does not call ``update_history``.

``FaceField.replace_packed`` replaces buffers without in-place writes into old
autograd leaves. IBM-specific packing remains a field responsibility rather
than an algorithm-specific collection of upper/lower face members. Existing
``pack`` / ``unpack`` behavior is unchanged.

Capture is detached by default. ``detach=False`` retains graph links for
differentiable tests or short replay chains. Restoration clones storage even
for detached checkpoints, preventing solver updates from modifying a saved
state.

Checkpoint restore validates the grid/configuration, geometry generations,
field registry and buffer layouts before changing fields. Checkpoints are
in-memory objects tied to their original grid, not portable restart files.
Hold the configuration fixed for replay; changing it requires a new checkpoint.
``validate(grid)`` performs the same compatibility checks without restoring
values or clearing caches. ``AlgorithmCheckpoint.validate(algorithm)`` also
checks the algorithm instance. Validation does not compare current field
values with saved values or inspect arbitrary mutable boundary/model objects.

External callers use ``FieldBindings`` for current-value tensor I/O and
``GridBase.freeze_field_registry()`` to reject unprepared field registration
during evaluation. See :ref:`external-field-io` for ownership and usage.

Algorithm checkpoints
---------------------

``algorithms.checkpoint.AlgorithmCheckpoint`` adds explicit iteration controls
to the field checkpoint. SIMPLE stores its initial/current residual history;
PIMPLE also stores its convergence flag. The diagnostic step counter is
included, but external diagnostic files and callbacks are not rolled back.
Replay adapters suppress external diagnostics during derivative evaluation.

The base algorithm captures only the common diagnostic counter. SIMPLE and
PIMPLE explicitly implement capture and restore for their own iteration
controls; PISO currently inherits the base implementation. No optional private
attribute names are inferred. ``IterationState`` copies residual mappings and
exposes read-only views; restore creates mutable dictionaries for the algorithm.

For fixed inputs, grid, time step and solver settings:

.. code-block:: python

   from gridfoam.algorithms.checkpoint import AlgorithmCheckpoint

   saved = AlgorithmCheckpoint.capture(algo)
   algo.step()
   first = AlgorithmCheckpoint.capture(algo)

   saved.restore(algo)
   # Reapply design inputs here if they changed between evaluations.
   algo.step()
   replay = AlgorithmCheckpoint.capture(algo)

Geometry and cache lifetime
---------------------------

The background grid owns connectivity and Cartesian geometry. IBM providers
own their selected surface entities and continuous boundary geometry:

* APIBM: selected ray/triangle hits, sides and near-boundary selections;
  continuous distances and intersection points.
* Shared wall queries: nearest-surface information, wall distance and normals.
  An AP-axis intersection distance is not automatically a turbulence wall
  distance.

``GridBase.geometry_revision`` changes after IBM updates, remeshing and device
moves. ``topology_revision`` additionally changes when background connectivity
is rebuilt. Even a remesh with unchanged counts invalidates old checkpoints.

Geometry providers call ``mark_geometry_changed`` after replacing or modifying
geometry and call ``invalidate_derived_caches``. Direct tensor edits require
both calls; ``mark_geometry_changed`` alone does not clear caches.
``AxisProjectedGrid.update_ib``, ``remesh`` and ``to`` already do both.
Checkpoints do not differentiate geometry or freeze a search trace.

``invalidate_derived_caches`` clears derived FV values without changing the
geometry generation. Restore always calls it, including when tensor values
are numerically equal: a cache created under ``no_grad`` or consumed by a
previous backward must not leak into a new graph. Caches remain scoped to a
single replay graph. Design applicators that change inputs without restoring
a checkpoint must invalidate the affected caches themselves.

In-place edits and aliases of exposed geometry tensors are not detected by
checkpoint validation. Unsupported edits must be prevented, or they will
leave stale checkpoints and caches in place.

GCIBM interpolation foothold
----------------------------

``fv.kernels.InterpolationStencil`` maps cell values to image-point samples
and has an exact transpose. Donor indices are discrete; weights stay
differentiable. It is not used by production assembly.

