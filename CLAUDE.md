# CLAUDE.md

## Project Overview
Video generation and editing system with audio capabilities. Testing infrastructure has been set up using Poetry and pytest.

## Commands
- Test: `poetry run test` or `poetry run tests`
- Test without coverage: `poetry run pytest --no-cov`
- Test specific markers: `poetry run pytest -m unit` or `poetry run pytest -m integration`
- Lint: TBD
- Type Check: TBD

## Testing Infrastructure
- Package Manager: Poetry
- Test Framework: pytest with coverage and mocking support
- Test Directory: `/tests` with unit and integration subdirectories
- Coverage Threshold: 80% (configurable in pyproject.toml)
- Test Markers: `unit`, `integration`, `slow`

## Notes
- Python 3.9+ required (due to accelerate dependency)
- Run `poetry install` to install all dependencies
- Coverage reports generated in `htmlcov/` directory and `coverage.xml`