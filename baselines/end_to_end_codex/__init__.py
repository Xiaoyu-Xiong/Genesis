"""End-to-end single-Codex-agent baseline."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from baselines.end_to_end_codex.runner import EndToEndBaselineConfig, run_end_to_end_suite

__all__ = ["EndToEndBaselineConfig", "run_end_to_end_suite"]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from baselines.end_to_end_codex.runner import EndToEndBaselineConfig, run_end_to_end_suite

    exports = {
        "EndToEndBaselineConfig": EndToEndBaselineConfig,
        "run_end_to_end_suite": run_end_to_end_suite,
    }
    return exports[name]
