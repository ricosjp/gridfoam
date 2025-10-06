CUDA_TAG = cu124

.PHONY: reset
reset:
	rm -r ./.venv || true
	rm uv.lock || true

.PHONY: install
install:
	uv sync --refresh --reinstall --extra ${CUDA_TAG}

.PHONY: dev-install
dev_install:
	uv sync --refresh --reinstall --extra ${CUDA_TAG} --group dev

.PHONY: lint
lint:
	uv run ruff check --output-format=full
	uv run ruff format --diff

.PHONY: cubion-test
cubion-test:
	cd src/cubion && cargo nextest run

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
	mkdir -p ./tests/outputs/benchmark/time/
	uv run pytest -v -m with_benchmark --benchmark-min-rounds=3 --benchmark-max-time=0.0001 --benchmark-save-data --benchmark-time-unit='ms' --benchmark-storage=./tests/outputs/benchmark/time --benchmark-autosave

.PHONY: profile
profile:
	mkdir -p ./tests/outputs/profile/time/
	mkdir -p ./tests/outputs/profile/memory/
	uv run pyinstrument -r html -o ./tests/outputs/profile/time/profile.html -m pytest -v -m with_profile
	uv run pytest -v -m with_profile --memray --memray-bin-path=./tests/outputs/profile/memory --memray-bin-prefix=gridgen
	uv run memray flamegraph -f ./tests/outputs/profile/memory/gridgen-tests-test_gridfoam-test_profile.py-test_gridgen_bunny_profile.bin
	uv run memray flamegraph -f ./tests/outputs/profile/memory/gridgen-tests-test_gridfoam-test_profile.py-test_gridgen_DrivAer_profile.bin

.PHONY: performance_check
performance_check: benchmark profile

