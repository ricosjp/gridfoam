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