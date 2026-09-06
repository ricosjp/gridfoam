# Torch extra: cpu, cu118, or cu124. Do not combine cpu with a CUDA extra.
CUDA_TAG ?= cu124

.PHONY: help
help:
	@echo "install / dev-install   uv sync (CUDA_TAG=$(CUDA_TAG))"
	@echo "lint                    basedpyright, ruff check, ruff format"
	@echo "cpu-test                default pytest (excludes slow/profile/benchmark)"
	@echo "slow-test               pytest -m slow"
	@echo "document                sphinx HTML"
	@echo "benchmark               resolution sweep under tests/profile"
	@echo "profile-time / profile-memory"
	@echo "experiment-gf-vs-of     OpenFOAM vs gridfoam slice comparison"
	@echo "experiment-re-vs-cd     sphere Re–Cd sweep"

.PHONY: reset
reset:
	rm -r ./.venv || true
	rm uv.lock || true

.PHONY: install
install:
	uv sync --refresh --reinstall --extra ${CUDA_TAG} --extra graphlow

.PHONY: dev-install
dev-install:
	uv sync --refresh --reinstall --extra ${CUDA_TAG} --extra graphlow --group dev

.PHONY: lint
lint:
	uv run basedpyright
	uv run ruff check --output-format=full
	uv run ruff format --diff

.PHONY: cpu-test
cpu-test:
	uv run pytest tests --cov=src --cov-report term-missing --durations 5

.PHONY: slow-test
slow-test:
	uv run pytest tests -m slow --cov=src --cov-report term-missing --durations 5

.PHONY: document
document:
	rm -rf docs/build || true
	rm -rf docs/source/api_reference/generated/ || true
	uv run sphinx-build docs/source docs/build -b html

.PHONY: benchmark
benchmark:
	uv run python tests/profile/benchmark_resolution.py gridfoam
	uv run python tests/profile/plot_cells_vs_time.py

.PHONY: profile-time
profile-time:
	mkdir -p ./tests/profile/outputs/time/
	GRIDFOAM_RUNTIME_TYPE_CHECKS=0 uv run pyinstrument -r html -o ./tests/profile/outputs/time/profile.html -m pytest -v -m profile

.PHONY: profile-memory
profile-memory:
	mkdir -p ./tests/profile/outputs/memory/
	GRIDFOAM_RUNTIME_TYPE_CHECKS=0 uv run pytest -v -m profile --memray --memray-bin-path=./tests/profile/outputs/memory --memray-bin-prefix=gridfoam
	# uv run memray flamegraph -f {bin_path}

.PHONY: experiment-gf-vs-of
experiment-gf-vs-of:
	GRIDFOAM_RUNTIME_TYPE_CHECKS=0 uv run python experiments/gf_vs_of/run.py

.PHONY: experiment-re-vs-cd
experiment-re-vs-cd:
	GRIDFOAM_RUNTIME_TYPE_CHECKS=0 uv run python experiments/re_vs_cd/run.py

.PHONY: performance_check
performance_check: benchmark profile-time profile-memory
