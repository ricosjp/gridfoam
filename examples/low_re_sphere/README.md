# Low-Reynolds Sphere Drag

This example compares sphere drag from CFD against the paper correlation
summarized by Clift et al., 1978

https://api.pageplace.de/preview/DT0400.9780429522871_A46452874/preview-9780429522871_A46452874.pdf

## Run gridfoam (Re sweep)

```bash
uv run python examples/low_re_sphere/gridfoam/run.py --re 0.5 1 2 5 10 100
```

Each Re result is written under `outputs/re_<Re>/`, and a merged summary is
written to `outputs/sweep_summary.csv`.

### Run OpenFOAM (Re sweep)

```bash
uv run python examples/low_re_sphere/of/run_sweep.py --re 0.5 1 2 5 10 100
```

This writes:

```text
examples/low_re_sphere/of/outputs/sweep_summary.csv
```

## Compare with paper Cd

After running one or both sweeps:

```bash
uv run python examples/low_re_sphere/compare.py
```
