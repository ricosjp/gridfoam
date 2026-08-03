.. _contributor-documentation:

Documentation
=============

Documentation sources live under ``docs/source`` and use reStructuredText.
Keep pages short near the top of the tree; put detail on deeper pages.

Detail by depth
---------------

Index pages
   One short paragraph and links to the next pages.

Overview pages
   Maps, responsibilities, and runtime flow without checklists or large
   excerpts.

Task and contract pages
   Steps, shapes, caveats, and testing checklists.

Python docstrings
   Public API behavior closest to the implementation.

Where content belongs
---------------------

``docs/source/user_guide``
   Installation, configuration, and running examples.

``docs/source/example_gallery``
   Case studies with committed figures and source references.

``docs/source/api_reference``
   Public Python API generated from explicit autosummary lists.

``docs/source/contributor_guide``
   Architecture overviews, setup, extension checklists, and project policies.

``docs/source/release_notes``
   Version history and migration notes.

Writing conventions
-------------------

* Write documentation and Python docstrings in English.
* Use NumPy-style Python docstrings.
* Document public attributes with an ``Attributes`` section and, for
  properties or class-level fields, a one-line attribute docstring so
  autosummary pages are not empty.
* State tensor shapes using the names used by the implementation, such as
  ``[C, k]`` and ``[F_patch, k]``.
* Distinguish current behavior from planned behavior explicitly.
* Link to a design concept instead of copying it into several task pages.
* Prefer references to stable classes or modules over exhaustive file trees,
  which become stale quickly.

Adding a page
-------------

#. Create an ``.rst`` file in the appropriate section.
#. Add a unique label at the top when the page may be referenced externally.
#. Add the page to a ``toctree``.
#. Use ``:doc:`` for document links and double backticks for code identifiers.
#. Build the documentation and resolve every warning introduced by the change.

Build locally
-------------

.. code-block:: console

   $ make document

The generated HTML is written to ``docs/build``. Open
``docs/build/index.html`` and check navigation, code blocks, diagrams, and
cross-references in addition to checking the command's exit status.

Keeping architecture documentation current
------------------------------------------

When a change moves responsibilities between packages, update
:doc:`architecture/packages`. When it changes a core data contract, update
:doc:`architecture/data_contracts`. When it changes dependency or extension
rules, update :doc:`architecture/guidelines`. Keep
:doc:`architecture/index` as a short overview. When a change adds a new
extension step, update the relevant page under :doc:`extending/index`.
