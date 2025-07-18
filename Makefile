.PHONY: reset
reset:
	rm -r ./.venv || true
	rm uv.lock || true

.PHONY: install
install:
	uv sync --refresh --reinstall

.PHONY: mypy
mypy:
	poetry run mypy src

.PHONY: lint
lint:
	uv run ruff check --output-format=full
	uv run ruff format --diff
	# uv run mypy src

.PHONY: cpu-test
cpu-test:
	uv run pytest tests --cov=src --cov-report term-missing --durations 5

.PHONY: gpu-test
gpu-test:
	uv run pytest tests -m with_device

.PHONY: document
document:
	rm -rf docs/build || true
	rm -rf docs/source/reference/generated || true
	rm -rf docs/source/tutorials || true
	rm docs/source/sg_execution_times.rst || true
	uv run sphinx-build -M html docs/source docs/build

.PHONY: benchmark
benchmark:
	uv run pytest -v -m with_benchmark --benchmark-min-rounds=3 --benchmark-max-time=0.0001 --benchmark-save-data --benchmark-time-unit='ms' --benchmark-storage=./tests/outputs/benchmark/time/ --benchmark-save=gridgen
	uv run pytest -v -m with_memray --memray --memray-bin-path=./tests/outputs/benchmark/memory --memray-bin-prefix=gridgen
	uv run memray flamegraph ./tests/outputs/benchmark/memory/gridgen-tests-test_gridfoam-benchmark-test_memory.py-test_grid_generation_process_bunny_benchmark_memory.bin

