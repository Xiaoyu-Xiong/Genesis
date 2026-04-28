# AGENTS.md - Genesis AI Agent Guide

Guide for AI coding assistants working with the Genesis physics simulation codebase.

## Execution Environment Rules

These rules are mandatory for this repository.

- **Never run Python outside Apptainer.** Any `python`, `python -m ...`, `uv run ...`, `uv sync`, `uv pip ...`, `pytest`, or other Python environment command must run **inside** Apptainer only.
- **Use the repository's standard Apptainer image unless the user explicitly says otherwise.** The current default image is `/ocean/projects/cis250078p/xxiong1/containers/genesis.sif`.
- **Never modify the host `.venv`.** Do not create, delete, recreate, sync, or repair `/jet/home/xxiong1/Genesis/.venv` from the host shell.
- **Git commands run outside Apptainer.** Use the host shell for `git status`, `git diff`, `git checkout`, `git commit`, `git restore`, and similar repository operations.
- **Host-shell work is read-only unless it is git.** Outside Apptainer, restrict actions to file inspection (`ls`, `cat`, `sed`, `rg`, `find`, `stat`) and git. Do not run Python tooling there, and do not run shell scripts that indirectly invoke repository Python from the host.
- **If execution context is unclear, stop and clarify the shell context before running commands that mutate environments or execute Python.**
- **For CPU-only or lightweight non-simulation tasks, run directly inside Apptainer.** Use the shortest correct command.
- **For GPU-dependent Genesis simulation, rendering, profiling, or long optimization tasks, default to `sbatch` on the `GPU-shared` partition with one `h100-80` GPU unless the user explicitly asks for a lighter local Apptainer smoke test.** Queueing can take time, so do not assume immediate execution.
- **For commands which require OpenAI API request**, the API response will take time based on OpenAI server loads, so do not assume immediate execution.
- **When the user says they are already inside Apptainer, give commands without an Apptainer prefix.** In that case, do not re-wrap commands with `apptainer exec`.
- **When the user asks for a command, prefer the shortest correct command.** Do not wrap a simple rerun in an unnecessarily complex script.

## Quick Start

```bash
# Setup (inside Apptainer only)
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu126  # or cpu/metal

# Run tests (inside Apptainer only)
uv run pytest tests/
uv run pytest tests/ -m required  # minimal set

# Run examples (inside Apptainer only)
uv run examples/tutorials/hello_genesis.py
```

## How to Run Tests

```bash
# All commands below are for Apptainer only.
uv run pytest tests/                      # All tests
uv run pytest tests/test_file.py          # Specific file
uv run pytest tests/ --backend=gpu        # GPU backend
uv run pytest tests/ -m required          # Required tests only
uv run pytest tests/ -m "not slow"        # Skip slow tests
```

See [TESTING.md](.github/contributing/TESTING.md) for details.

## How to Contribute

### PR Title Prefixes

- `[BUG FIX]` - Non-breaking bug fixes
- `[FEATURE]` - New functionality
- `[MISC]` - Minor changes (docs, typos)
- `[CHANGING]` - Behavior changes
- `[BREAKING]` - Breaking API changes

### Before Submitting

1. Install pre-commit hooks: `pre-commit install`
2. Run required tests inside Apptainer: `uv run pytest -m required tests/`
3. Link to related issue in PR description

See [PULL_REQUESTS.md](.github/contributing/PULL_REQUESTS.md) for details.

## Formatting & Lint

Genesis uses **ruff** for linting and formatting (via pre-commit):

```bash
# Install hooks (auto-runs on commit)
pre-commit install

# Manual run
pre-commit run --all-files
```

**Rules:**
- Line length: 120 characters
- Format: ruff-format (black-compatible)
- Lint: ruff-check

See [CODING_CONVENTIONS.md](.github/contributing/CODING_CONVENTIONS.md) for code style.

## When to Ask a Human

Ask for clarification when:

- **Ambiguous requirements** - Multiple valid interpretations exist
- **Breaking changes** - Changes that affect public APIs or behavior
- **Architecture decisions** - New solvers, major refactors, new entity types
- **Performance trade-offs** - When optimization conflicts with readability
- **Test failures** - Unclear why tests fail or how to fix them
- **Cross-solver coupling** - Changes affecting multiple physics solvers

Do NOT ask when:
- Standard bug fixes with clear reproduction steps
- Documentation updates
- Adding tests for existing functionality
- Code style fixes flagged by linters

## Reference Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](.github/contributing/ARCHITECTURE.md) | Project structure, solvers, entities |
| [TESTING.md](.github/contributing/TESTING.md) | Testing guide and fixtures |
| [CODING_CONVENTIONS.md](.github/contributing/CODING_CONVENTIONS.md) | Code style and patterns |
| [EXAMPLES.md](.github/contributing/EXAMPLES.md) | Examples reference |
| [PULL_REQUESTS.md](.github/contributing/PULL_REQUESTS.md) | PR guidelines |
| [agent/README.md](agent/README.md) | Legacy `agent/` pipeline index, entry points, and pipeline-specific maintenance rules |
