# Torch extra: cpu, cu118, or cu124. Do not combine cpu with a CUDA extra.
CUDA_TAG ?= cu124

# Performance tools run CPU and CUDA sequentially by default.
PERF_DEVICES ?= cpu cuda
PERF_ARGS ?=
BENCHMARK_ARGS ?=
PROFILE_ARGS ?=

.PHONY: help
help:
	@echo "install / dev-install   uv sync (CUDA_TAG=$(CUDA_TAG))"
	@echo "lint                    basedpyright, ruff check, ruff format"
	@echo "cpu-test                default pytest (excludes slow/profile/benchmark)"
	@echo "slow-test               pytest -m slow"
	@echo "document                sphinx HTML"
	@echo "benchmark               CPU/CUDA sweep + plot; reuse cached OpenFOAM"
	@echo "benchmark-openfoam      force OpenFOAM resolution sweep"
	@echo "profile-time / profile-memory  CPU/CUDA reports in outputs/{time,memory}"
	@echo "  PERF_DEVICES=cpu|cuda  PERF_ARGS='--steps 2 --cpu-threads 4'"
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
	GRIDFOAM_RUNTIME_TYPE_CHECKS=0 uv run python -m tests.profile.benchmark_resolution --devices $(PERF_DEVICES) $(PERF_ARGS) $(BENCHMARK_ARGS)

.PHONY: benchmark-openfoam
benchmark-openfoam:
	GRIDFOAM_RUNTIME_TYPE_CHECKS=0 uv run python -m tests.profile.benchmark_resolution openfoam $(PERF_ARGS) $(BENCHMARK_ARGS)

.PHONY: profile-time
profile-time:
	GRIDFOAM_RUNTIME_TYPE_CHECKS=0 uv run python -m tests.profile.run_profiles time --devices $(PERF_DEVICES) $(PERF_ARGS) $(PROFILE_ARGS)

.PHONY: profile-memory
profile-memory:
	GRIDFOAM_RUNTIME_TYPE_CHECKS=0 uv run python -m tests.profile.run_profiles memory --devices $(PERF_DEVICES) $(PERF_ARGS) $(PROFILE_ARGS)

.PHONY: experiment-gf-vs-of
experiment-gf-vs-of:
	GRIDFOAM_RUNTIME_TYPE_CHECKS=0 uv run python experiments/gf_vs_of/run.py

.PHONY: experiment-re-vs-cd
experiment-re-vs-cd:
	GRIDFOAM_RUNTIME_TYPE_CHECKS=0 uv run python experiments/re_vs_cd/run.py

.PHONY: performance_check
performance_check: benchmark profile-time profile-memory
