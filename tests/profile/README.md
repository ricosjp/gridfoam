# CPU and CUDA performance tools

Run from the repository root, using an environment with the `dev` dependency
group installed:

```sh
make benchmark
make profile-time
make profile-memory
```

All three commands run CPU and CUDA sequentially in separate processes. CUDA
requires a CUDA-enabled PyTorch installation and an available GPU. If CUDA is
unavailable, the default runs CPU only and prints the reason. Explicitly
requesting CUDA alone fails when it is unavailable.

## Choosing the workload

The shared configuration is `tests/profile/data/config.yaml`. Profiling uses
its resolution and duration; benchmarking sweeps `(5, 2, 2)`, `(10, 4, 4)`,
`(20, 8, 8)`, and `(40, 16, 16)` unless overridden. The largest CPU case can
take a long time. Use a short run to check the tools first:

```sh
make benchmark profile-time profile-memory \
  PERF_ARGS="--steps 2 --resolution '5 2 2' --cpu-threads 2" \
  BENCHMARK_ARGS="--without-openfoam"

make profile-time PERF_DEVICES=cpu
make profile-memory PERF_DEVICES=cuda
make benchmark BENCHMARK_ARGS="--resolution '5 2 2' --resolution '10 4 4'"
```

| Make variable | Options |
| --- | --- |
| `PERF_DEVICES` | `cpu cuda` (default), `cpu`, or `cuda` |
| `PERF_ARGS` | Shared `--config PATH`, `--steps N` or `--end-time T`, `--resolution 'NX NY NZ'`, `--cpu-threads N` |
| `BENCHMARK_ARGS` | `--without-openfoam`, `--skip-existing`, `--csv PATH`; repeated `--resolution` values |
| `PROFILE_ARGS` | `--output DIRECTORY`, `--write-fields`, `--native`, `--history-entries N` |

The original YAML is never modified. Effective configurations are saved with
the profiles. CPU thread settings apply to PyTorch in both CPU and CUDA runs;
the default preserves PyTorch's thread count. `PERF_DEVICES` does not change
the CPU-only OpenFOAM baseline.

## Outputs

| Command | Default output under `tests/profile/` |
| --- | --- |
| `benchmark` | `benchmark_results/resolution_scaling.csv`, `resolution_scaling-latest.csv`, `cells_vs_time.png` |
| `profile-time` | `outputs/time/{cpu,cuda}/profile.html` |
| `profile-memory` | `outputs/memory/{cpu,cuda}/host.bin`, `host.html`, `memory.png` |

Each device's profile directory also contains `config.json` and `summary.json`.
The parent `outputs/time/summary.json` or `outputs/memory/summary.json` lists
the devices completed in the latest invocation and their results. Reports for
devices omitted from a run may remain on disk; use the parent summary to
identify the current comparison. `--output` replaces the parent directory,
so use different directories for time and memory runs when retaining both.

Benchmark CSV rows include device, configuration hash, duration, CPU thread
count, cell count, hardware name, PyTorch version, setup time, and solve time.
CPU and CUDA rows cannot overwrite each other. Matching rows are replaced;
different configurations and durations remain in the historical CSV. The plot
uses only the latest invocation, plus OpenFOAM rows with matching resolution
and duration. Legacy gridfoam rows without a known device are retained as
`unknown` and excluded from the latest comparison.

By default gridfoam is measured again. `--skip-existing` reuses rows matching
device, resolution, configuration, duration, and thread count. The cache does
not detect source-code or hardware changes; omit this flag after changing
either, or use a separate `--csv` path.

## Reading the measurements

All gridfoam tools call the same workload: build the mesh and algorithm, then
execute a fixed number of steps. They exclude imports, CUDA context creation,
early stopping, and post-processing. VTU export is disabled unless
`--write-fields` is passed to profiling, in which case each device writes its
own `fields.vtu` and reports `write_s` separately.

`setup_s` covers mesh and algorithm initialization. `solve_s` covers the step
loop. `elapsed_s` is setup + solve + optional export. CUDA is synchronized at
measurement boundaries, so these times include completion of queued GPU work.
Use `solve_s / n_steps` for the mean step time; the legacy CSV column
`time_per_step_s` includes setup as well.

Use benchmark timings for speed comparisons. Pyinstrument and especially
memory tracing add overhead. Pyinstrument's HTML shows Python call stacks and
GPU waits, not individual CUDA kernel timings.

Memory outputs distinguish host memory from GPU memory:

- `host.html` is a Memray allocation flamegraph. `host.bin` can also be read
  with `uv run memray stats PATH`. Add `PROFILE_ARGS="--native"` for C/C++
  allocation stacks, at additional tracing cost.
- `memory.png` shows host RSS and Memray-tracked heap. CUDA runs add PyTorch
  allocated memory, allocator-reserved memory, and cumulative peak allocation.
- CUDA `summary.json` includes counters sampled after initialization and each
  step, with peaks that also capture allocations freed between samples.
- CUDA runs save `cuda-initialization.pickle` and `cuda-final.pickle`. Open
  these in the [PyTorch memory visualizer](https://pytorch.org/memory_viz).
  History is bounded to 100,000 entries by default; use
  `PROFILE_ARGS="--history-entries 200000"` for a longer history. Initialization
  is saved separately so that its allocation history is preserved.

Reserved CUDA memory includes the allocator's reusable cache. PyTorch counters
and snapshots do not cover allocations made directly outside its allocator,
such as some CUDA library/context memory. Memray heap size and process RSS
also measure different things. Memory-profile timings include per-step
synchronization and history recording and should not be used as benchmark
timings.

## OpenFOAM baseline

`make benchmark` reuses cached OpenFOAM rows with matching resolution,
`endTime`, and `deltaT`, and measures missing rows if OpenFOAM and its case are
available. If unavailable, it prints a message and continues with gridfoam.
Use `BENCHMARK_ARGS="--without-openfoam"` to omit the baseline entirely.
`make benchmark-openfoam` explicitly reruns OpenFOAM and requires its case and
commands. Both the mesh resolution and `endTime` edits are restored on exit,
including when execution fails.

The OpenFOAM baseline keeps its own case settings. Its `elapsed_time.txt`
measures the scope selected by `Allrun`, which includes more than solver
iterations in the bundled case. Treat it as a separate end-to-end baseline;
it is not equivalent to gridfoam's `solve_s` or an identical discretization.

## Script layout

| Module | Responsibility |
| --- | --- |
| `case.py` | Shared CLI options, configuration, workload, and timing boundaries |
| `benchmark_resolution.py` | Resolution/device orchestration and benchmark workers |
| `benchmark_results.py` | CSV identities, updates, and legacy result migration |
| `benchmark_openfoam.py` | OpenFOAM case execution and result parsing |
| `run_profiles.py` | Sequential CPU/CUDA profile orchestration |
| `profile_case.py` | One device's time or memory capture and reports |
| `plot_cells_vs_time.py` | Benchmark comparison plot |
| `plot_simple_step_breakdown.py` | Optional SIMPLE step breakdown from time HTML |

The Make targets call Python modules rather than wrapping pytest in profilers.
Run `uv run python -m tests.profile.run_profiles --help` or
`uv run python -m tests.profile.benchmark_resolution --help` for CLI details.
`test_motorBike.py` remains an optional pytest entry point for the shared
workload, excluded from the default correctness suite.
