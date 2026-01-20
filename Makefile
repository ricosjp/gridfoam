CUDA_TAG = cu124

.PHONY: reset
reset:
	rm -r ./.venv || true
	rm uv.lock || true

.PHONY: install
install:
	uv sync --refresh --reinstall --extra ${CUDA_TAG}

.PHONY: dev-install
dev-install:
	uv sync --refresh --reinstall --extra ${CUDA_TAG} --group dev

.PHONY: lint
lint:
	uv run ruff check --output-format=full
	uv run ruff format --diff

.PHONY: cpu-test
cpu-test:
	uv run pytest tests --cov=src --cov-report term-missing --durations 5

.PHONY: gpu-test
gpu-test:
	uv run pytest tests -m with_device --cov=src --cov-report term-missing --durations 5

.PHONY: document
document:
	rm -rf docs/build || true
	rm -rf docs/source/tutorials || true
	rm docs/source/sg_execution_times.rst || true
	uv run sphinx-build docs/source docs/build -b html

.PHONY: benchmark
benchmark:
	mkdir -p ./tests/outputs/benchmark/time/
	uv sync --refresh --reinstall --group benchmark
	uv run pytest -v -m with_benchmark --benchmark-min-rounds=3 --benchmark-save-data --benchmark-time-unit='ms' --benchmark-storage=./tests/outputs/benchmark --benchmark-autosave
	uv run python visualization/benchmark.py

.PHONY: profile-time
profile-time:
	mkdir -p ./tests/outputs/profile/time/
	uv run pyinstrument -r html -o ./tests/outputs/profile/time/profile.html -m pytest -v -m with_profile

.PHONY: profile-memory
profile-memory:
	mkdir -p ./tests/outputs/profile/memory/
	uv run pytest -v -m with_profile --memray --memray-bin-path=./tests/outputs/profile/memory --memray-bin-prefix=gridgen
	uv run memray flamegraph -f ./tests/outputs/profile/memory/gridgen-tests-test_gridfoam-test_profile.py-test_gridgen_bunny_profile.bin
	uv run memray flamegraph -f ./tests/outputs/profile/memory/gridgen-tests-test_gridfoam-test_profile.py-test_gridgen_DrivAer_profile.bin

.PHONY: performance_check
performance_check: benchmark profile

