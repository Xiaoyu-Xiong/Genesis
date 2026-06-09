"""Prompt module mode switch.

The default prompt mode uses compact prompts plus Planner-dispatched SimDebug
cards. Set CODE_AGENT_PROMPT_MODE=legacy before process startup to run the
pre-card prompt snapshots for ablation experiments.
"""

from __future__ import annotations

import importlib
import os
import sys


def prompt_mode() -> str:
    return os.environ.get("CODE_AGENT_PROMPT_MODE", "cards").strip().lower() or "cards"


if prompt_mode() == "legacy":
    for _name in ("common", "ipc", "worker", "planner", "critic", "opt"):
        sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"code_agent.prompts_legacy.{_name}")
