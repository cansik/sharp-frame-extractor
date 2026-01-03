.PHONY: fmt fmt-check lint autoformat

default: fmt-check

# Code Style
fmt:
	uv run --group lint ruff format .

fmt-check:
	uv run --group lint ruff format --check .

lint:
	uv run --group lint ruff check .

autoformat:
	uv run --group lint ruff format . && uv run --group lint ruff check --fix .

# Deployment
build:
	rm -rf dist build *.egg-info && uv build