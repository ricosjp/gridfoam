# Low-Reynolds Sphere Drag

This example compares sphere drag from CFD against the paper correlation
summarized by Ramírez (2017), using Cheng (2009):

```text
Cd = 24/Re * (1 + 0.27 Re)^0.43 + 0.47[1 - exp(-0.04 Re^0.38)]
```

## Run gridfoam (Re sweep)

```bash
uv run python examples/low_re_sphere/gridfoam/run.py --re 0.5 1 2 5 10
```

Each Re result is written under `outputs/re_<Re>/`, and a merged summary is
written to `outputs/sweep_summary.csv`.

### Run OpenFOAM (Re sweep)

```bash
uv run python examples/low_re_sphere/of/run_sweep.py --re 0.5 1 2 5 10
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
