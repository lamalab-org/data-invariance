.PHONY: install lint format test check

install:
	uv sync --group dev
	uv run pre-commit install

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

check: lint test
