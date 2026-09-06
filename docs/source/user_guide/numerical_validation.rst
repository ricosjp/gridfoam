.. _user-numerical-validation:

Numerical validation
====================

Linear-field reconstruction checks consistency only. Convergence on curved
fields and boundedness at discontinuities are regression tests in
``tests/integration/fv/``. Criteria live in those files.

Manufactured Poisson
--------------------

``test_spatial_convergence.py`` solves
:math:`-\nabla^2 q = 2\pi^2\sin(\pi x)\sin(\pi y)` on uniform and
centrally refined octrees. Cell-value errors must show order greater than
the test threshold. Face fluxes, including hanging faces, must decrease;
the tests do not claim second-order interface gradients, wall shear, or
forces.

Scalar pulse advection
----------------------

``test_advection_bounds.py`` advects a compact pulse across a hanging
interface with Euler and deferred TVD. Upwind and TVD schemes must stay
bounded and conserve mass; linear interpolation must overshoot (negative
control). The check is for that mesh, CFL, and time scheme only.

Run the tests
-------------

.. code-block:: console

   $ python -m pytest tests/integration/fv/test_spatial_convergence.py tests/integration/fv/test_advection_bounds.py -q

Temporal-order and harmonic-diffusion tests are separate. IBM accuracy,
force convergence, smooth multidimensional advection, and remesh
conservation are not covered here.
