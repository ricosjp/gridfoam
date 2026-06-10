# Low-Reynolds Cube Drag

This example compares cube drag from CFD against the piecewise correlation
proposed by Wang et al. (2025), based on free-settling experiments of cubes.

For each Reynolds-number range, the comparison script uses:

```text
Cd = 24/Re * (1 + p1 Re^p2) + p3 Re^p4 / (Re + p5)
```

with parameters from Wang et al. (2025), Table 4.

## Run gridfoam (Re sweep)

```bash
uv run python examples/low_re_cube/gridfoam/run.py --re 0.5 1 2 5 10 100
```

Each Re result is written under `outputs/re_<Re>/`, and a merged summary is
written to `outputs/sweep_summary.csv`.

### Run OpenFOAM (Re sweep)

```bash
uv run python examples/low_re_cube/of/run_sweep.py --re 0.5 1 2 5 10 100
```

This writes:

```text
examples/low_re_cube/of/outputs/sweep_summary.csv
```

## Compare with paper Cd

After running one or both sweeps:

```bash
uv run python examples/low_re_cube/compare.py
```
