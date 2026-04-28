from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from code_agent.configs import CONFIGS

CodexSandbox = Literal["read-only", "workspace-write"]
DEFAULT_CODEX_PATH = "/jet/home/xxiong1/.vscode-server/extensions/openai.chatgpt-26.422.62136-linux-x64/bin/linux-x86_64/codex"


@dataclass(slots=True, frozen=True)
class CodexExecRequest:
    """Non-interactive `codex exec` request prepared by orchestration."""

    role: str
    prompt: str
    output_jsonl_path: Path
    final_message_path: Path
    cwd: Path = Path("/jet/home/xxiong1/Genesis")
    sandbox: CodexSandbox = "read-only"
    model: str | None = None
    output_schema_path: Path | None = None
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


@dataclass(slots=True, frozen=True)
class CodexResult:
    """Compatibility result for the early logs_dir-based runner shape."""

    role: str
    returncode: int
    duration_sec: float
    jsonl_path: Path
    final_message_path: Path
    stdout_path: Path
    stderr_path: Path

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.final_message_path.exists()


def resolve_codex_binary(codex_bin: str = "codex") -> str | None:
    if Path(codex_bin).exists():
        return codex_bin
    return shutil.which(codex_bin) or (DEFAULT_CODEX_PATH if Path(DEFAULT_CODEX_PATH).exists() else None)


def build_codex_exec_command(request: CodexExecRequest, *, resolved_codex: str | None = None) -> list[str]:
    if request.sandbox not in ("read-only", "workspace-write"):
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
    command.extend(request.extra_args)
    command.append(request.prompt)
    return command


def run_codex_exec(
    request: CodexExecRequest | None = None,
    *,
    role: str | None = None,
    prompt: str | None = None,
    workdir: Path | None = None,
    logs_dir: Path | None = None,
    sandbox: CodexSandbox | None = None,
    output_schema: Path | None = None,
    model: str | None = None,
    timeout_sec: float | None = None,
) -> CodexExecResult | CodexResult:
    """Run Codex in batch mode and persist stdout JSON events as JSONL.

    Prefer passing `CodexExecRequest`. The keyword-only form is kept for the
    initial logs_dir-based adapter and returns `CodexResult`.
    """

    if request is None:
        if role is None or prompt is None or workdir is None or logs_dir is None:
            raise TypeError("Either request or role/prompt/workdir/logs_dir must be provided")
        logs_dir.mkdir(parents=True, exist_ok=True)
        legacy_request = CodexExecRequest(
            role=role,
            prompt=prompt,
            cwd=workdir,
            sandbox=sandbox or "read-only",
            model=model,
            output_schema_path=output_schema,
            output_jsonl_path=logs_dir / f"codex_{role}.jsonl",
            final_message_path=logs_dir / f"codex_{role}.final.md",
            timeout_sec=timeout_sec,
        )
        result = _run_codex_exec_request(legacy_request)
        return CodexResult(
            role=result.role,
            returncode=result.exit_code if result.exit_code is not None else 127,
            duration_sec=result.duration_sec,
            jsonl_path=Path(result.output_jsonl_path),
            final_message_path=Path(result.final_message_path),
            stdout_path=Path(result.output_jsonl_path),
            stderr_path=Path(result.stderr_path) if result.stderr_path else logs_dir / f"codex_{role}.stderr",
        )

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
