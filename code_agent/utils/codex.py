from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from code_agent.configs import CONFIGS

CodexSandbox = Literal["read-only", "workspace-write", "danger-full-access"]
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CODEX_PATH: str | None = None


@dataclass(slots=True, frozen=True)
class CodexExecRequest:
    """Non-interactive `codex exec` request prepared by a code-agent caller."""

    role: str
    prompt: str
    output_jsonl_path: Path
    final_message_path: Path
    cwd: Path = field(default_factory=lambda: DEFAULT_REPO_ROOT)
    sandbox: CodexSandbox = "read-only"
    model: str | None = None
    output_schema_path: Path | None = None
    image_paths: tuple[Path, ...] = ()
    codex_bin: str = "codex"
    ask_for_approval: str = CONFIGS.codex.ask_for_approval
    timeout_sec: float | None = None
    extra_args: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class CodexExecResult:
    """Structured invocation result returned without hiding failed Codex calls."""

    role: str
    success: bool
    exit_code: int | None
    duration_sec: float
    command: list[str]
    cwd: str
    sandbox: str
    output_jsonl_path: str
    final_message_path: str
    output_schema_path: str | None
    codex_version: str | None
    error_type: str | None = None
    error_message: str | None = None
    stderr_path: str | None = None
    timed_out: bool = False
    started_at_unix: float = field(default_factory=time.time)
    ended_at_unix: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_codex_binary(codex_bin: str = "codex") -> str | None:
    if Path(codex_bin).exists():
        return codex_bin
    if resolved := shutil.which(codex_bin):
        return resolved
    if DEFAULT_CODEX_PATH and Path(DEFAULT_CODEX_PATH).exists():
        return DEFAULT_CODEX_PATH
    return None


def build_codex_exec_command(request: CodexExecRequest, *, resolved_codex: str | None = None) -> list[str]:
    if request.sandbox not in ("read-only", "workspace-write", "danger-full-access"):
        raise ValueError(f"Unsupported Codex sandbox: {request.sandbox}")

    command = [
        resolved_codex or request.codex_bin,
        "exec",
        "--cd",
        str(request.cwd),
        "--sandbox",
        request.sandbox,
        "--json",
        "--output-last-message",
        str(request.final_message_path),
    ]
    if request.model:
        command.extend(["--model", request.model])
    if request.output_schema_path is not None:
        command.extend(["--output-schema", str(request.output_schema_path)])
    for image_path in request.image_paths:
        command.extend(["--image", str(image_path)])
    command.extend(request.extra_args)
    if request.image_paths:
        command.append("--")
    command.append(request.prompt)
    return command


def run_codex_exec(request: CodexExecRequest) -> CodexExecResult:
    """Run Codex in batch mode and persist stdout JSON events as JSONL.

    Callers build an explicit request so output paths and execution policy stay visible at the callsite.
    """
    return _run_codex_exec_request(request)


def _run_codex_exec_request(request: CodexExecRequest) -> CodexExecResult:
    started = time.time()
    resolved_codex = resolve_codex_binary(request.codex_bin)
    command = build_codex_exec_command(request, resolved_codex=resolved_codex)
    jsonl_path = request.output_jsonl_path
    final_path = request.final_message_path
    stderr_path = jsonl_path.with_suffix(jsonl_path.suffix + ".stderr")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    if resolved_codex is None:
        message = f"Codex executable not found on PATH: {request.codex_bin}"
        _write_error_outputs(jsonl_path, final_path, request.role, "codex_not_found", message)
        ended = time.time()
        return _result(
            request,
            command,
            started,
            ended,
            None,
            None,
            "codex_not_found",
            message,
            stderr_path,
        )

    codex_version = _read_codex_version(resolved_codex)
    timed_out = False
    exit_code: int | None = None
    error_type: str | None = None
    error_message: str | None = None

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        try:
            process = subprocess.Popen(
                command,
                cwd=request.cwd,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
                encoding="utf-8",
            )
            stdout, _ = process.communicate(timeout=request.timeout_sec)
            jsonl_file.write(stdout)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            error_type = "timeout"
            error_message = f"Codex invocation timed out after {request.timeout_sec} seconds"
            process.kill()
            stdout, _ = process.communicate()
            jsonl_file.write(stdout)
            exit_code = process.returncode
            _append_jsonl_error(jsonl_file, request.role, error_type, error_message)
            if not final_path.exists():
                final_path.write_text(f"{error_type}: {error_message}\n", encoding="utf-8")
        except OSError as exc:
            exit_code = None
            error_type = "codex_launch_failed"
            error_message = str(exc)
            _append_jsonl_error(jsonl_file, request.role, error_type, error_message)
            final_path.write_text(f"{error_type}: {error_message}\n", encoding="utf-8")

    ended = time.time()
    if exit_code != 0 and error_type is None:
        error_type = "codex_exec_failed"
        error_message = f"Codex exited with status {exit_code}"

    return _result(
        request,
        command,
        started,
        ended,
        exit_code,
        codex_version,
        error_type,
        error_message,
        stderr_path,
        timed_out=timed_out,
    )


def _result(
    request: CodexExecRequest,
    command: list[str],
    started: float,
    ended: float,
    exit_code: int | None,
    codex_version: str | None,
    error_type: str | None,
    error_message: str | None,
    stderr_path: Path,
    *,
    timed_out: bool = False,
) -> CodexExecResult:
    return CodexExecResult(
        role=request.role,
        success=exit_code == 0,
        exit_code=exit_code,
        duration_sec=ended - started,
        command=command,
        cwd=str(request.cwd),
        sandbox=request.sandbox,
        output_jsonl_path=str(request.output_jsonl_path),
        final_message_path=str(request.final_message_path),
        output_schema_path=str(request.output_schema_path) if request.output_schema_path else None,
        codex_version=codex_version,
        error_type=error_type,
        error_message=error_message,
        stderr_path=str(stderr_path),
        timed_out=timed_out,
        started_at_unix=started,
        ended_at_unix=ended,
    )


def _read_codex_version(codex_bin: str) -> str | None:
    try:
        completed = subprocess.run(
            [codex_bin, "--version"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    version = (completed.stdout or completed.stderr).strip()
    return version or None


def _write_error_outputs(path: Path, final_path: Path, role: str, error_type: str, message: str) -> None:
    event = {
        "type": "error",
        "role": role,
        "error_type": error_type,
        "message": message,
    }
    path.write_text(json.dumps(event, ensure_ascii=False) + "\n", encoding="utf-8")
    final_path.write_text(f"{error_type}: {message}\n", encoding="utf-8")


def _append_jsonl_error(jsonl_file, role: str, error_type: str, message: str) -> None:
    event = {
        "type": "error",
        "role": role,
        "error_type": error_type,
        "message": message,
    }
    jsonl_file.write(json.dumps(event, ensure_ascii=False) + "\n")
