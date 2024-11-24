# Create environment with uv, including dependencies and dev dependencies
create-virtualenv:
	uv sync

# Run the tests
test: create-virtualenv
	echo "Running tests"
	uv run pytest tests/

# Run the tests with coverage
test-coverage: create-virtualenv
	echo "Running test coverage"
	uv run pytest --cov=src --cov-report term-missing tests/

# Run full test-suite (needs root permissions)
test-full: create-virtualenv
	echo "Running tests"
	sudo -E env PATH="$(PATH)" uv run pytest tests/

# Run full test-suite with coverage (needs root permissions)
test-full-coverage: create-virtualenv
	echo "Running full test coverage"
	sudo -E env PATH="$(PATH)" uv run pytest --cov=src --cov-report term-missing tests/
	sudo chown -R $(USER):$(USER) .coverage