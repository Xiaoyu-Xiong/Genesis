from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from code_agent.utils.codex import CodexExecResult

WorkerRole = Literal["scene", "body", "action", "rendering"]


@dataclass(slots=True, frozen=True)
class WorkerSpec:
    role: WorkerRole
    target_file: str
    required_export: str
    responsibility: str
    prompt_body: str


@dataclass(slots=True)
class WorkerDispatchResult:
    role: WorkerRole
    ok: bool
    target_path: Path
    codex_result: CodexExecResult
    worker_report: dict[str, object] | None = None
    error_message: str | None = None
