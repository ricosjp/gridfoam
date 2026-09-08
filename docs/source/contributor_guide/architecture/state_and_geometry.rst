.. _state-and-geometry:

State, replay, and geometry contracts
=====================================

The M0 foundation preserves existing YAML, examples, and public solver APIs.
APIBM continues to support thin sheets and baffles. The future GCIBM solver
targets closed surfaces; the interpolation prototype does not itself validate
surface closure or implement a GCIBM flow solver.

Package ownership
-----------------

Fluxel owns surface input, grid construction, search, and IBM geometry.
Gridfoam owns numerical state, discretization, physical models, and solve/VJP
APIs. The phlower integration module belongs in phlower, which depends on
gridfoam. Gridfoam does not depend on phlower or PhlowerTrainer. The existing
``phlower-tensor`` dependency is a separate package.

Differentiable blocks and design inputs
---------------------------------------

``core.state.TensorState`` is an ordered mapping of named tensor blocks. It
supports heterogeneous shapes and preserves the supplied tensor graph, so
boundary inputs, geometry tensors, and neural-network parameters need not be
flattened into one scalar design variable. Integer selections and connectivity
belong in metadata, outside the differentiable blocks.

``as_tuple`` and ``from_tuple`` bridge to autograd VJPs with a strict layout.
Key order, shape, dtype and device must agree; vector addition must never
silently broadcast or truncate state blocks.
``validate_layout`` checks this contract against a named mapping without
constructing a state or copying tensors. Addition and checkpoint replacement
use this validation directly.

* ``clone()`` copies storage and preserves the graph.
* ``detach()`` discards the graph but shares storage.
* ``checkpoint()`` copies storage and discards the graph.

This differs from ``FlowState.clone()`` on the experimental
``feature/adjoint_for_algorithms`` branch, which also detaches. When adopting
that branch, use ``checkpoint()`` where a detached primal snapshot is intended.

Boundary conditions, material/model parameters, and solver settings are explicit
inputs to replay. They are not implicitly captured as evolving field state.
Reapply them for every evaluation, including after restoring a checkpoint.
The first trainer integration optimizes inlet conditions; learned model
corrections can later use the same tensor-input contract.

Field checkpoints
-----------------

``core.checkpoint.GridCheckpoint`` captures all registered cell and face fields,
including an added temperature field, model fields, and scratch fields. Strong
field references keep the grid's weak registry alive. This complete replay
checkpoint is distinct from the minimal independent unknown vector used by a
fixed-point adjoint; selecting that vector is part of the M1 step-map adapter.

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
state. ``with_values`` replaces the tensor blocks with the same layout for
VJP evaluation.

Checkpoint restore validates the grid/configuration, geometry generations,
field registry and buffer layouts before changing fields. Checkpoints are
in-memory objects tied to their original grid, not portable restart files.
Hold the configuration fixed for replay; changing it requires a new checkpoint.

Algorithm checkpoints
---------------------

``algorithms.checkpoint.AlgorithmCheckpoint`` adds explicit iteration controls
to the field checkpoint. SIMPLE stores its initial/current residual history;
PIMPLE also stores its convergence flag. The diagnostic step counter is
included, but external diagnostic files and callbacks are not rolled back.
Replay adapters should suppress external diagnostics during derivative
evaluation. Future models with evolving non-field state must explicitly
extend the checkpoint contract.

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

The M0 tests compare all saved tensor blocks and iteration controls after
replaying SIMPLE, PISO and PIMPLE with an additional diffusion scalar. Euler
and BDF2 configurations are covered. This verifies replay; it does not yet
provide an algorithm-level adjoint or a Trainer adapter.

Geometry and cache lifetime
---------------------------

The background grid owns connectivity and Cartesian geometry. IBM providers
own their selected surface entities and continuous boundary geometry:

* APIBM: selected ray/triangle hits, sides and near-boundary selections;
  continuous distances and intersection points.
* GCIBM: fluid/ghost and donor selections; continuous boundary projections,
  image points and interpolation weights.
* Shared wall queries: nearest-surface information, wall distance and normals.
  An AP-axis intersection distance is not automatically a turbulence wall
  distance.

In M2, search outputs form a fixed trace for each derivative evaluation while
continuous geometry is recomputed from input vertices. The M0 checkpoint
contract does not differentiate geometry or freeze a trace on its own.

``GridBase.geometry_revision`` changes after IBM updates, remeshing and device
moves. ``topology_revision`` additionally changes when background connectivity
is rebuilt. Even a remesh with unchanged counts invalidates old checkpoints.

Geometry providers call ``mark_geometry_changed`` after replacing or modifying
geometry and call ``invalidate_derived_caches``. Direct tensor edits require
both calls; ``mark_geometry_changed`` alone does not clear caches.
``AxisProjectedGrid.update_ib``, ``remesh`` and ``to`` already do both. General
geometry updates and remapping inside an adjoint trajectory are outside M0.

``invalidate_derived_caches`` clears derived FV values without changing the
geometry generation. Restore always calls it, including when tensor values
are numerically equal: a cache created under ``no_grad`` or consumed by a
previous backward must not leak into a new graph. Caches remain scoped to a
single replay graph. Design applicators that change inputs without restoring
a checkpoint must invalidate the affected caches themselves.

M2 must provide a geometry update API that combines tensor replacement,
generation changes and cache invalidation. Its design must also address
in-place edits and aliases of exposed geometry tensors: prevent unsupported
edits or detect them and reject stale checkpoints/caches. An update API alone
does not enforce this. M0 still relies on the explicit notification contract.

GCIBM interpolation prototype
-----------------------------

``fv.kernels.ghost_interpolation.InterpolationStencil`` represents the linear
map from cell values to image-point samples. Donor indices are discrete;
weights remain differentiable. ``apply`` and ``transpose_apply`` support
scalar, vector and tensor fields, negative weights, repeated donor indices,
and empty image sets. Missing donors must never be encoded as negative
indices, which would silently select cells from the end of a tensor.

M0 keeps interpolation separate from the face-based LDU matrix. A manufactured
Poisson problem on a closed interval demonstrates ghost constraints of the
form ``u_ghost + interpolate(u) = 2 * u_wall``, non-neighbour couplings, and a
non-symmetric system requiring the correct transpose. Its shape derivative
is checked against autograd and finite differences. The small test constructs
a dense matrix only as a reference; production assembly remains unchanged.

M4 must decide whether to eliminate ghost unknowns, represent additional sparse
couplings, or compose a matrix-free operator. Both primal and transpose paths
must include the interpolation. Surface closure, donor-search robustness,
conservation and physical boundary conditions still require M4 validation.
