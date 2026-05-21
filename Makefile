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
	uv run basedpyright
	uv run ruff check --output-format=full
	uv run ruff format --diff

.PHONY: cpu-test
cpu-test:
	uv run pytest tests --cov=src --cov-report term-missing --durations 5

.PHONY: gpu-test
gpu-test:
	uv run pytest tests -m with_device --cov=src --cov-report term-missing --durations 5

# For headless CI: install xvfb and run ``xvfb-run make document``.
.PHONY: document
document:
	rm -rf docs/build || true
	rm -rf docs/source/api_reference/generated/ || true
	rm -rf docs/source/example_gallery/auto_examples || true
	rm docs/source/sg_execution_times.rst || true
	uv run sphinx-build docs/source docs/build -b html

.PHONY: benchmark
benchmark:
	mkdir -p ./tests/outputs/benchmark
	uv run pytest -v -m benchmark \
		--benchmark-min-rounds=3 \
		--benchmark-save-data \
		--benchmark-time-unit=ms \
		--benchmark-json=./tests/outputs/benchmark/latest.json \
		--benchmark-storage=./tests/outputs/benchmark \
		--benchmark-autosave

.PHONY: profile-time
profile-time:
	mkdir -p ./tests/profile/outputs/time/
	GRIDFOAM_RUNTIME_TYPE_CHECKS=0 uv run pyinstrument -r html -o ./tests/profile/outputs/time/profile.html -m pytest -v -m profile

.PHONY: profile-memory
profile-memory:
	mkdir -p ./tests/profile/outputs/memory/
	GRIDFOAM_RUNTIME_TYPE_CHECKS=0 uv run pytest -v -m profile --memray --memray-bin-path=./tests/profile/outputs/memory --memray-bin-prefix=gridfoam
	# uv run memray flamegraph -f {bin_path}

.PHONY: experiment
experiment:
	GRIDFOAM_RUNTIME_TYPE_CHECKS=0 uv run python -m experiments.run --config experiments/config.yml

.PHONY: performance_check
performance_check: benchmark profile-time profile-memory
