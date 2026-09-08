# Numerical experiments

Run commands from the repository root in the development environment. Each
experiment uses its own configuration; check its device and workload before
running. Most operator and external-flow configurations default to CUDA.

| Experiment | Command | Configuration and output |
| --- | --- | --- |
| Scalar advection | `uv run experiments/operator/advection/advection.py` | Adjacent `config.yaml`; temperature VTU files and log in `outputs/` |
| Scalar diffusion | `uv run experiments/operator/diffusion/diffusion.py` | Adjacent `config.yaml`; temperature VTU files and log in `outputs/` |
| Gradient reconstruction | `uv run experiments/operator/grad/grad.py` | Adjacent `config.yaml`; analytical-field comparisons in `outputs/` |
| Manufactured Poisson | `uv run experiments/operator/poisson/poisson.py --n-leaf-refinements 1 2` | Renders `config.j2`; error CSV and convergence plot in `outputs/`; `--save-vtu` adds fields |
| gridfoam/OpenFOAM slices | `make experiment-gf-vs-of` | `gf_vs_of/parameters.yml` selects cases under `examples/` |
| Reynolds number/drag | `uv run experiments/re_vs_cd/run.py --solver gridfoam --no-mlflow` | `re_vs_cd/data/parameters.yml` and templates; `--cases` and `--re` limit the sweep |

The Poisson experiment solves a manufactured Dirichlet problem on a box.
`operator/poisson/config.yml` is a static example of the template at leaf
refinement 1; the script reads `config.j2`. Its convergence plot measures
cell-value error and does not validate immersed-boundary flux or force
accuracy. Focused immersed-boundary regressions live in
`tests/integration/fv/test_immersed_dirichlet.py`.

Manual scalar experiments assemble all terms before `equation(field, matrix)`
applies immersed Dirichlet cell constraints, and read the solver's
`SolveResult.solution`. Scalar data has shape `[C]`, with scalar YAML values
such as `0.0`. Advection honors the prescribed boundary condition on inflow
and outflow; `inlet_outlet` owns flow-direction switching. See the
[data contracts](../docs/source/contributor_guide/architecture/data_contracts.rst).

The drag template uses SIMPLE, so its pressure tolerance is configured under
`p`. PISO/PIMPLE cases may also configure `pFinal`. OpenFOAM comparisons
require a sourced OpenFOAM environment; their case dictionaries remain in
OpenFOAM's own format.

Regenerate simulation outputs and plots after changing numerical behavior;
existing figures and MLflow runs record the code/settings used at execution
time. CPU/CUDA timing and memory comparisons use the separate
[performance tools](../tests/profile/README.md).
