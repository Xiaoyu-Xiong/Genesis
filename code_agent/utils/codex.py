from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

from code_agent.assets.builtin_guard import builtin_asset_denied_roots
from code_agent.configs import CONFIGS

CodexSandbox = Literal["read-only", "workspace-write", "danger-full-access"]
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CODEX_PATH: str | None = None
USAGE_LIMIT_MARKERS = (
    "usage limit",
    "purchase more credits",
    "try again",
)


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
    reasoning_effort: str | None = CONFIGS.codex.reasoning_effort
    service_tier: Literal["fast", "standard"] | None = CONFIGS.codex.service_tier
    timeout_sec: float | None = None
    extra_args: tuple[str, ...] = ()
    hide_builtin_assets: bool = CONFIGS.codex.hide_builtin_assets_from_agents
    writable_roots: tuple[Path, ...] = ()


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
    if request.reasoning_effort:
        command.extend(["-c", f'model_reasoning_effort="{request.reasoning_effort}"'])
    cli_service_tier = _codex_cli_service_tier(request.service_tier)
    if cli_service_tier:
        command.extend(["-c", f'service_tier="{cli_service_tier}"'])
    if request.service_tier == "fast":
        command.extend(["-c", "features.fast_mode=true"])
    if request.output_schema_path is not None:
        command.extend(["--output-schema", str(request.output_schema_path)])
    for image_path in request.image_paths:
        command.extend(["--image", str(image_path)])
    command.extend(request.extra_args)
    command.append("-")
    return _wrap_with_asset_sandbox(request, command)


def _codex_cli_service_tier(service_tier: Literal["fast", "standard"] | None) -> str | None:
    """Map public code-agent names to the service tier tokens accepted by Codex CLI."""

    if service_tier == "standard":
        return None
    return service_tier


def run_codex_exec(request: CodexExecRequest) -> CodexExecResult:
    """Run Codex in batch mode and persist stdout JSON events as JSONL.

    Callers build an explicit request so output paths and execution policy stay visible at the callsite.
    """
    return _run_codex_exec_request(request)


def _run_codex_exec_request(request: CodexExecRequest) -> CodexExecResult:
    request = _normalize_request_paths(request)
    started = time.time()
    jsonl_path = request.output_jsonl_path
    final_path = request.final_message_path
    stderr_path = jsonl_path.with_suffix(jsonl_path.suffix + ".stderr")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_codex = resolve_codex_binary(request.codex_bin)
    if resolved_codex is not None and request.hide_builtin_assets and builtin_asset_denied_roots():
        bwrap_bin = shutil.which("bwrap")
        if bwrap_bin is None:
            message = "bwrap is required to hide Genesis built-in assets from Codex agents, but it is not on PATH."
            _write_error_outputs(jsonl_path, final_path, request.role, "asset_sandbox_unavailable", message)
            ended = time.time()
            return _result(
                request,
                [request.codex_bin],
                started,
                ended,
                None,
                None,
                "asset_sandbox_unavailable",
                message,
                stderr_path,
            )
    command = build_codex_exec_command(request, resolved_codex=resolved_codex)
    final_path.unlink(missing_ok=True)

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
                stdin=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                start_new_session=True,
            )
            stdout, _ = process.communicate(input=request.prompt, timeout=request.timeout_sec)
            jsonl_file.write(stdout)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            error_type = "timeout"
            error_message = f"Codex invocation timed out after {request.timeout_sec} seconds"
            _kill_process_tree(process)
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
        classified_type, classified_message = _classify_codex_failure(jsonl_path=jsonl_path, stderr_path=stderr_path)
        error_type = classified_type or "codex_exec_failed"
        error_message = classified_message or f"Codex exited with status {exit_code}"
        if not final_path.exists():
            final_path.write_text(f"{error_type}: {error_message}\n", encoding="utf-8")

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


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    """Terminate a Codex wrapper and any child binary it launched."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (AttributeError, ProcessLookupError, OSError):
        try:
            process.kill()
        except ProcessLookupError:
            pass


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


def _normalize_request_paths(request: CodexExecRequest) -> CodexExecRequest:
    """Resolve paths before native mesh libraries can perturb process cwd.

    Suite cases run in threads. Some native mesh-processing bindings briefly
    change the process-global cwd, so all Codex IO paths must be absolute before
    mkdir/open/subprocess calls happen.
    """

    return replace(
        request,
        cwd=_repo_path(request.cwd),
        output_jsonl_path=_repo_path(request.output_jsonl_path),
        final_message_path=_repo_path(request.final_message_path),
        output_schema_path=None if request.output_schema_path is None else _repo_path(request.output_schema_path),
        image_paths=tuple(_repo_path(path) for path in request.image_paths),
        writable_roots=tuple(_repo_path(path) for path in request.writable_roots),
    )


def _repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (DEFAULT_REPO_ROOT / path).resolve()


def _wrap_with_asset_sandbox(request: CodexExecRequest, command: list[str]) -> list[str]:
    if not request.hide_builtin_assets:
        return command
    denied_roots = tuple(path for path in builtin_asset_denied_roots() if path.exists())
    if not denied_roots:
        return command
    bwrap_bin = shutil.which("bwrap")
    if bwrap_bin is None:
        return command

    wrapper = [
        bwrap_bin,
        "--ro-bind",
        "/",
        "/",
        "--dev-bind",
        "/dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
    ]
    for root in _asset_sandbox_writable_roots(request, denied_roots=denied_roots):
        wrapper.extend(["--bind", str(root), str(root)])
    for root in denied_roots:
        wrapper.extend(["--tmpfs", str(root)])
    wrapper.extend(["--chdir", str(request.cwd)])
    return [*wrapper, *command]


def _asset_sandbox_writable_roots(request: CodexExecRequest, *, denied_roots: tuple[Path, ...]) -> tuple[Path, ...]:
    roots = [
        request.output_jsonl_path.parent,
        request.final_message_path.parent,
        *request.writable_roots,
    ]
    codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).expanduser()
    if codex_home.exists():
        roots.append(codex_home)
    resolved: list[Path] = []
    for root in roots:
        path = root.resolve()
        if not path.exists():
            continue
        if any(_is_relative_to(path, denied) or _is_relative_to(denied, path) for denied in denied_roots):
            continue
        if any(path == existing or _is_relative_to(path, existing) for existing in resolved):
            continue
        resolved = [existing for existing in resolved if not _is_relative_to(existing, path)]
        resolved.append(path)
    return tuple(resolved)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


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


def _classify_codex_failure(*, jsonl_path: Path, stderr_path: Path) -> tuple[str | None, str | None]:
    messages: list[str] = []
    for path in (jsonl_path, stderr_path):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            message = _json_event_message(line) or line
            messages.append(message)
            lower = message.lower()
            if all(marker in lower for marker in ("usage limit", "try again")) or any(
                marker in lower for marker in USAGE_LIMIT_MARKERS[:2]
            ):
                return "codex_usage_limit", message
    combined = "\n".join(messages).lower()
    if all(marker in combined for marker in ("usage limit", "try again")) or any(
        marker in combined for marker in USAGE_LIMIT_MARKERS[:2]
    ):
        return "codex_usage_limit", _first_nonempty(messages)
    return None, None


def _json_event_message(line: str) -> str | None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict):
        return None
    message = event.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    error = event.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return None


def _first_nonempty(messages: list[str]) -> str | None:
    for message in messages:
        if message.strip():
            return message.strip()
    return None
