.. _contributor-configuration:

Configuration
=============

Simulation YAML is validated by Pydantic models in
``src/gridfoam/meta/config.py``. Serialized choices are ``StrEnum`` members in
``src/gridfoam/meta/enums.py``. Runtime code should consume the validated model
rather than repeatedly reading raw dictionaries.

Adding a configuration option
-----------------------------

#. Add the field to the smallest owning Pydantic model.
#. Give it an explicit type and a safe default when the option is backward
   compatible.
#. Add a field or model validator only when type validation is insufficient.
#. Update a representative YAML file or template when users should see the
   option.
#. Add parsing tests for valid, default, and invalid values.

Configuration models are generally frozen. Do not rely on mutating them during
a simulation; derive runtime state in the owning algorithm or model instead.

Adding a selectable implementation
----------------------------------

Features selected by name normally require:

#. A value in ``meta/enums.py``.
#. An enum-typed field in ``meta/config.py``.
#. A factory branch or scheme dispatch-table entry.
#. The implementation itself.
#. A YAML parsing test and an implementation test.

Keep serialized enum values stable. Renaming an enum member changes accepted
YAML because ``StrEnum`` values are used directly.

Scheme keys
-----------

Finite-volume scheme dictionaries use OpenFOAM-style string keys. The
configuration validator normalizes whitespace around commas, so code should
use the validated keys rather than duplicating normalization.

Field configuration
-------------------

``CellField`` looks up its condition by field name when it is constructed. It
then initializes internal data, creates boundary conditions through
``boundaries.factory``, and registers itself on the grid. A change to field
naming must account for optional phase-qualified names such as ``U.water``;
use helpers in ``core/name.py`` instead of concatenating names manually.

Testing
-------

Configuration tests belong under ``tests/unit/meta/`` or the existing
``tests/unit/test_meta_config.py`` area. Test a complete bundled YAML in an
integration test only when the goal is to validate wiring across configuration
sections.
