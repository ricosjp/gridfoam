# Low-Reynolds Circular Cylinder Drag

This example compares cylinder drag from CFD against the paper correlation
used in Elghannay & El Hasadi (2023), namely the Sucker-Brauwer (1975) fit:

```text
Cd = 6.8/Re^0.89 + 1.96/Re^0.5 + 1.18 - 1/(2500/Re + Re/1100)
```

https://mdp.omu.edu.ly/journals/index.php/mjer/article/view/59

## Run gridfoam (Re sweep)

```bash
uv run python examples/low_re_cylinder/gridfoam/run.py --re 0.5 1 2 5 10 100
```

Each Re result is written under `outputs/re_<Re>/`, and a merged summary is
written to `outputs/sweep_summary.csv`.

### Run OpenFOAM (Re sweep)

```bash
uv run python examples/low_re_cylinder/of/run_sweep.py --re 0.5 1 2 5 10 100
```

This writes:

```text
examples/low_re_cylinder/of/outputs/sweep_summary.csv
```

## Compare with paper Cd

After running one or both sweeps:

```bash
uv run python examples/low_re_cylinder/compare.py
```
