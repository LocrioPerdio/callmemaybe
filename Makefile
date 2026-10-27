.PHONY: install run debug clean lint lint-strict test

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m src --help

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

lint:
	uv run flake8 src
	uv run mypy src

lint-strict:
	uv run flake8 src
	uv run mypy --strict src