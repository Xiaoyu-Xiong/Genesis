# AGENTS.md - Genesis AI Agent Guide

Guide for AI coding assistants working with the Genesis physics simulation codebase.

## Execution Environment Rules

These rules are mandatory for this repository.

- **Use the repository uv environment to run python script.** Do not dircectly run `python`, `python -m ...`, `pytest`, but run `uv run --no-sync ...`, `uv sync`, `uv pip ...` instead.
- **Be careful when mutating `.venv`.** Do not recreate or bulk repair the environment unless the user asks for environment work.
- **Use the host shell for git.** Use ordinary `git status`, `git diff`, `git checkout`, `git commit`, `git restore`, and similar repository operations.
- **If execution context is unclear, stop and clarify before running commands that mutate environments or launch expensive simulations.**
- **Use the dedicated local GPU by default.** Run GPU-capable Genesis simulation, rendering, profiling, optimization, tests, and examples directly on the local GPU. Use CPU only when the user asks, the GPU is unavailable, or the task is explicitly CPU-only.
- **For commands which require OpenAI API request**, the API response will take time based on OpenAI server loads, so do not assume immediate execution.
- **When the user asks for a command, prefer the shortest correct command.** Do not wrap a simple rerun in an unnecessarily complex script.
- **Put memory caps on long WSL runs.** Launch overnight suites, mesh/XML asset batches, and other memory-heavy background
  jobs under a process-tree memory limit. Prefer `systemd-run --user` with `MemoryMax=...`; fall back to `ulimit` only
  when cgroups are unavailable. Record the run root, log path, service/unit or PID, and memory cap in the response.
- **Preserve `uv` and CUDA library paths in non-interactive/systemd environments.** User systemd services and other
  non-interactive shells may not inherit the interactive shell `PATH` or `LD_LIBRARY_PATH`. When launching suites or
  generated simulations through `systemd-run`, pass an explicit `PATH` containing the directory that owns `uv`, or invoke
  `uv` by absolute path. On WSL GPU runs, also pass the interactive CUDA library path (for example
  `/home/xxyfh/Genesis/.venv/cuda-12.8/lib:/usr/lib/wsl/lib`) so Genesis does not load an incompatible system
  `libcuda.so`. Otherwise nested execution may fail as inconclusive with `uv: command not found` or die during CUDA
  initialization.

## Quick Start

```bash
# Setup
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu128  # or cpu/metal

# Run tests
uv run --no-sync pytest tests/
uv run --no-sync pytest tests/ -m required  # minimal set

# Run examples
uv run --no-sync examples/tutorials/hello_genesis.py
```

## How to Run Tests

```bash
uv run --no-sync pytest tests/                      # All tests
uv run --no-sync pytest tests/test_file.py          # Specific file
uv run --no-sync pytest tests/ --backend=gpu        # GPU backend, default for GPU-capable validation
uv run --no-sync pytest tests/ -m required          # Required tests only
uv run --no-sync pytest tests/ -m "not slow"        # Skip slow tests
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
2. Run required tests: `uv run pytest -m required tests/`
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
