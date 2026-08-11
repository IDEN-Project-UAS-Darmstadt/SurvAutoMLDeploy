# Default recipe
default:
    @just --list

# Display available commands
help:
    @just --list

# Run Ruff and markdownlint
lint:
    ruff check src
    markdownlint . --ignore node_modules

# Auto-format Python code and Jupyter notebooks
format:
    ruff check --fix src
    ruff format src
    markdownlint . -i -q --fix || true

# Verify code formatting without applying changes
format-check:
    ruff format --check src
    ruff check src

# Run all checks
check-all: lint format-check
    @echo "✓ All checks passed!"