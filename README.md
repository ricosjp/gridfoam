# Gridfoam: Differentiable Fluid Solver of Hierarchical Grid

gridfoam is a differentiable finite-volume solver for incompressible flow on
Fluxel hierarchical grids. It provides SIMPLE, PISO, and PIMPLE algorithms,
immersed boundaries, and PyTorch gradients through linear solves.

Start with the [CPU cavity quickstart](docs/source/user_guide/quickstart.rst).
See [configuration](docs/source/user_guide/configuration.rst) for YAML settings
and [running cases](docs/source/user_guide/running_cases.rst) for bundled examples.

- [`examples/`](examples/) contains runnable flow, motion, and optimization cases.
- [`assets/`](assets/README.md) contains reusable Jinja configuration templates.
- [`experiments/`](experiments/README.md) contains numerical and OpenFOAM comparisons.
- [`tests/profile/`](tests/profile/README.md) contains CPU/CUDA performance tools.

For development, see the [setup guide](docs/source/contributor_guide/development_setup.rst)
and [data contracts](docs/source/contributor_guide/architecture/data_contracts.rst).
Build the HTML documentation with `make document`; the result is in `docs/build/`.
