from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from code_agent.assets.builtin_guard import builtin_asset_denied_roots
from code_agent.configs import CONFIGS
from code_agent.utils.local_execution import build_local_execution_env

CodexSandbox = Literal["read-only", "workspace-write", "danger-full-access"]
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CODEX_PATH: str | None = None
USAGE_LIMIT_MARKERS = (
    "usage limit",
    "purchase more credits",
    "try again",
)
CODEX_ACCOUNT_FILE_ENV = "CODE_AGENT_CODEX_ACCOUNT_FILE"
CODEX_ACCOUNTS_ENV = "CODE_AGENT_CODEX_ACCOUNTS"
CODEX_QUOTA_WAIT_ENV = "CODE_AGENT_CODEX_QUOTA_WAIT"
CODEX_QUOTA_PROBE_INTERVAL_ENV = "CODE_AGENT_CODEX_QUOTA_PROBE_INTERVAL_SEC"
CODEX_QUOTA_PROBE_INITIAL_DELAY_ENV = "CODE_AGENT_CODEX_QUOTA_PROBE_INITIAL_DELAY_SEC"
CODEX_QUOTA_PROBE_TIMEOUT_ENV = "CODE_AGENT_CODEX_QUOTA_PROBE_TIMEOUT_SEC"
CODEX_QUOTA_PROBE_PROMPT_ENV = "CODE_AGENT_CODEX_QUOTA_PROBE_PROMPT"
CODEX_QUOTA_PROBE_MODEL_ENV = "CODE_AGENT_CODEX_QUOTA_PROBE_MODEL"


@dataclass(slots=True, frozen=True)
class _CodexAccount:
    """One pre-authenticated Codex account profile.

    `codex_home` should point at a directory that has already been logged in with
    `CODEX_HOME=/path/to/profile codex login`. `profile` maps to Codex CLI
    `--profile`; it is optional and mainly useful when a CODEX_HOME contains
    several config profiles.
    """

    name: str
    codex_home: Path | None = None
    profile: str | None = None
    top_level_args: tuple[str, ...] = ()


@dataclass(slots=True)
class _CodexAccountState:
    quota_limited: bool = False
    auth_failed: bool = False
    last_error: str | None = None
    last_checked_at_unix: float | None = None


_CODEX_ACCOUNT_LOCK = threading.RLock()
_CODEX_ACCOUNT_STATES: dict[str, _CodexAccountState] = {}
_ACTIVE_CODEX_ACCOUNT_NAME: str | None = None


def _reset_codex_account_state_for_tests() -> None:
    global _ACTIVE_CODEX_ACCOUNT_NAME
    with _CODEX_ACCOUNT_LOCK:
        _CODEX_ACCOUNT_STATES.clear()
        _ACTIVE_CODEX_ACCOUNT_NAME = None


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
    codex_top_level_args: tuple[str, ...] = ()
    ask_for_approval: str = CONFIGS.codex.ask_for_approval
    reasoning_effort: str | None = CONFIGS.codex.reasoning_effort
    service_tier: Literal["fast", "standard"] | None = CONFIGS.codex.service_tier
    timeout_sec: float | None = None
    extra_args: tuple[str, ...] = ()
    hide_builtin_assets: bool = CONFIGS.codex.hide_builtin_assets_from_agents
    writable_roots: tuple[Path, ...] = ()
    codex_home: Path | None = None
    codex_account_name: str | None = None


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
    codex_account_name: str | None = None
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
        *request.codex_top_level_args,
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
    request = _normalize_request_paths(request)
    if not _codex_quota_wait_enabled():
        return _run_codex_exec_request_once(request)

    accounts = _configured_codex_accounts()
    attempted_accounts: set[str] = set()
    last_result: CodexExecResult | None = None
    while True:
        account = _select_codex_account(accounts, exclude=attempted_accounts)
        if account is None:
            if last_result is not None and not _has_codex_quota_limited_account(accounts):
                return last_result
            account = _wait_for_codex_quota_recovery(request, accounts)
            attempted_accounts.clear()

        result = _run_codex_exec_request_once(_request_for_codex_account(request, account))
        last_result = result
        if result.error_type == "codex_usage_limit":
            _mark_codex_account_quota_limited(account, result.error_message)
            attempted_accounts.add(account.name)
            _append_codex_quota_event(
                request,
                {
                    "event": "account_usage_limited",
                    "account": account.name,
                    "message": result.error_message,
                    "will_try_next_account": len(attempted_accounts) < len(accounts),
                },
            )
            continue
        if result.error_type == "codex_auth_failed" and len(accounts) > 1:
            _mark_codex_account_auth_failed(account, result.error_message)
            attempted_accounts.add(account.name)
            _append_codex_quota_event(
                request,
                {
                    "event": "account_auth_failed",
                    "account": account.name,
                    "message": result.error_message,
                    "will_try_next_account": len(attempted_accounts) < len(accounts),
                },
            )
            continue
        if result.success:
            _mark_codex_account_success(account)
        return result


def _codex_quota_wait_enabled() -> bool:
    raw = os.environ.get(CODEX_QUOTA_WAIT_ENV)
    if raw is not None:
        return raw.strip().lower() not in {"0", "false", "off", "no"}
    return bool(CONFIGS.codex.quota_auto_wait)


def _configured_codex_accounts() -> tuple[_CodexAccount, ...]:
    accounts = _codex_accounts_from_file() or _codex_accounts_from_env()
    if not accounts:
        accounts = (_default_codex_account(),)
    return _dedupe_codex_accounts(accounts)


def _codex_accounts_from_file() -> tuple[_CodexAccount, ...]:
    raw_path = os.environ.get(CODEX_ACCOUNT_FILE_ENV)
    if not raw_path:
        return ()
    path = Path(raw_path).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_accounts: Any = payload.get("accounts") if isinstance(payload, dict) else payload
    if not isinstance(raw_accounts, list):
        raise ValueError(f"{CODEX_ACCOUNT_FILE_ENV} must point to a JSON object/list with accounts")
    return tuple(
        account
        for index, item in enumerate(raw_accounts)
        if (account := _codex_account_from_mapping(item, default_index=index)) is not None
    )


def _codex_accounts_from_env() -> tuple[_CodexAccount, ...]:
    raw = os.environ.get(CODEX_ACCOUNTS_ENV, "").strip()
    if not raw:
        return ()
    accounts: list[_CodexAccount] = []
    for index, token in enumerate(re.split(r"[;,]", raw)):
        item = token.strip()
        if not item:
            continue
        if "=" in item:
            name, spec = item.split("=", 1)
            name = name.strip() or f"account_{index + 1}"
            spec = spec.strip()
        else:
            spec = item
            name = Path(spec).expanduser().name or f"account_{index + 1}"
        if spec.startswith("profile:"):
            accounts.append(_CodexAccount(name=name, profile=spec.removeprefix("profile:").strip() or None))
        else:
            accounts.append(_CodexAccount(name=name, codex_home=Path(spec).expanduser()))
    return tuple(accounts)


def _codex_account_from_mapping(item: object, *, default_index: int) -> _CodexAccount | None:
    if not isinstance(item, dict):
        return None
    raw_home = item.get("codex_home") or item.get("home") or item.get("CODEX_HOME")
    codex_home = Path(str(raw_home)).expanduser() if raw_home else None
    profile = item.get("profile")
    profile_text = str(profile).strip() if profile is not None else None
    raw_args = item.get("top_level_args") or item.get("codex_top_level_args") or ()
    top_level_args = tuple(str(arg) for arg in raw_args) if isinstance(raw_args, list) else ()
    name = str(item.get("name") or item.get("id") or "").strip()
    if not name:
        if codex_home is not None:
            name = codex_home.name
        elif profile_text:
            name = profile_text
        else:
            name = f"account_{default_index + 1}"
    return _CodexAccount(
        name=name,
        codex_home=codex_home,
        profile=profile_text,
        top_level_args=top_level_args,
    )


def _default_codex_account() -> _CodexAccount:
    raw_home = os.environ.get("CODEX_HOME")
    return _CodexAccount(
        name=os.environ.get("CODE_AGENT_CODEX_ACCOUNT_NAME", "default"),
        codex_home=Path(raw_home).expanduser() if raw_home else None,
    )


def _dedupe_codex_accounts(accounts: tuple[_CodexAccount, ...]) -> tuple[_CodexAccount, ...]:
    seen: dict[str, int] = {}
    deduped: list[_CodexAccount] = []
    for account in accounts:
        base_name = account.name or "account"
        count = seen.get(base_name, 0)
        seen[base_name] = count + 1
        name = base_name if count == 0 else f"{base_name}_{count + 1}"
        top_level_args = account.top_level_args
        if account.profile:
            top_level_args = ("--profile", account.profile, *top_level_args)
        deduped.append(replace(account, name=name, top_level_args=top_level_args))
    return tuple(deduped) or (_default_codex_account(),)


def _select_codex_account(accounts: tuple[_CodexAccount, ...], *, exclude: set[str]) -> _CodexAccount | None:
    with _CODEX_ACCOUNT_LOCK:
        ordered = _ordered_codex_accounts(accounts)
        for account in ordered:
            if account.name in exclude:
                continue
            state = _CODEX_ACCOUNT_STATES.get(account.name)
            if state is not None and (state.quota_limited or state.auth_failed):
                continue
            return account
    return None


def _ordered_codex_accounts(accounts: tuple[_CodexAccount, ...]) -> tuple[_CodexAccount, ...]:
    if _ACTIVE_CODEX_ACCOUNT_NAME is None:
        return accounts
    for index, account in enumerate(accounts):
        if account.name == _ACTIVE_CODEX_ACCOUNT_NAME:
            return (*accounts[index:], *accounts[:index])
    return accounts


def _state_for_codex_account(account: _CodexAccount) -> _CodexAccountState:
    state = _CODEX_ACCOUNT_STATES.get(account.name)
    if state is None:
        state = _CodexAccountState()
        _CODEX_ACCOUNT_STATES[account.name] = state
    return state


def _has_codex_quota_limited_account(accounts: tuple[_CodexAccount, ...]) -> bool:
    with _CODEX_ACCOUNT_LOCK:
        return any(
            (state := _CODEX_ACCOUNT_STATES.get(account.name)) is not None and state.quota_limited
            for account in accounts
        )


def _mark_codex_account_quota_limited(account: _CodexAccount, message: str | None) -> None:
    with _CODEX_ACCOUNT_LOCK:
        state = _state_for_codex_account(account)
        state.quota_limited = True
        state.auth_failed = False
        state.last_error = message
        state.last_checked_at_unix = time.time()


def _mark_codex_account_auth_failed(account: _CodexAccount, message: str | None) -> None:
    with _CODEX_ACCOUNT_LOCK:
        state = _state_for_codex_account(account)
        state.auth_failed = True
        state.last_error = message
        state.last_checked_at_unix = time.time()


def _mark_codex_account_success(account: _CodexAccount) -> None:
    global _ACTIVE_CODEX_ACCOUNT_NAME
    with _CODEX_ACCOUNT_LOCK:
        state = _state_for_codex_account(account)
        state.quota_limited = False
        state.auth_failed = False
        state.last_error = None
        state.last_checked_at_unix = time.time()
        _ACTIVE_CODEX_ACCOUNT_NAME = account.name


def _wait_for_codex_quota_recovery(request: CodexExecRequest, accounts: tuple[_CodexAccount, ...]) -> _CodexAccount:
    started = time.time()
    _append_codex_quota_event(
        request,
        {
            "event": "quota_pause_started",
            "accounts": [account.name for account in accounts],
            "probe_interval_sec": _codex_quota_probe_interval_sec(),
        },
    )
    initial_delay = _codex_quota_probe_initial_delay_sec()
    if initial_delay > 0:
        _append_codex_quota_event(request, {"event": "quota_probe_sleep", "sleep_sec": initial_delay})
        time.sleep(initial_delay)

    while True:
        for account in accounts:
            result = _probe_codex_account_quota(request, account)
            event = {
                "event": "quota_probe_result",
                "account": account.name,
                "success": result.success,
                "error_type": result.error_type,
                "message": result.error_message,
                "duration_sec": result.duration_sec,
            }
            if result.success:
                _mark_codex_account_success(account)
                _append_codex_quota_event(
                    request,
                    {
                        **event,
                        "event": "quota_recovered",
                        "paused_sec": time.time() - started,
                    },
                )
                return account
            if result.error_type == "codex_usage_limit":
                _mark_codex_account_quota_limited(account, result.error_message)
            elif result.error_type == "codex_auth_failed":
                _mark_codex_account_auth_failed(account, result.error_message)
            _append_codex_quota_event(request, event)

        interval = _codex_quota_probe_interval_sec()
        _append_codex_quota_event(request, {"event": "quota_probe_sleep", "sleep_sec": interval})
        if interval > 0:
            time.sleep(interval)


def _probe_codex_account_quota(request: CodexExecRequest, account: _CodexAccount) -> CodexExecResult:
    probe_root = Path(
        os.environ.get(
            "CODE_AGENT_CODEX_QUOTA_PROBE_DIR",
            str(Path(tempfile.gettempdir()) / "code-agent-codex-quota"),
        )
    ).expanduser()
    probe_root.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", account.name)[:80] or "account"
    stamp = int(time.time() * 1000)
    probe_request = CodexExecRequest(
        role=f"quota_probe_{safe_name}",
        prompt=os.environ.get(CODEX_QUOTA_PROBE_PROMPT_ENV, CONFIGS.codex.quota_probe_prompt),
        cwd=DEFAULT_REPO_ROOT,
        sandbox="read-only",
        model=_codex_quota_probe_model(request),
        output_jsonl_path=probe_root / f"{safe_name}_{stamp}.jsonl",
        final_message_path=probe_root / f"{safe_name}_{stamp}.final.txt",
        codex_bin=request.codex_bin,
        codex_top_level_args=request.codex_top_level_args,
        reasoning_effort=None,
        service_tier=request.service_tier,
        timeout_sec=_codex_quota_probe_timeout_sec(),
        hide_builtin_assets=False,
    )
    return _run_codex_exec_request_once(_request_for_codex_account(_normalize_request_paths(probe_request), account))


def _codex_quota_probe_model(request: CodexExecRequest) -> str | None:
    return os.environ.get(CODEX_QUOTA_PROBE_MODEL_ENV) or CONFIGS.codex.quota_probe_model or request.model


def _codex_quota_probe_interval_sec() -> float:
    return _float_env(CODEX_QUOTA_PROBE_INTERVAL_ENV, CONFIGS.codex.quota_probe_interval_sec)


def _codex_quota_probe_initial_delay_sec() -> float:
    return _float_env(CODEX_QUOTA_PROBE_INITIAL_DELAY_ENV, CONFIGS.codex.quota_probe_initial_delay_sec)


def _codex_quota_probe_timeout_sec() -> float:
    return _float_env(CODEX_QUOTA_PROBE_TIMEOUT_ENV, CONFIGS.codex.quota_probe_timeout_sec)


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return max(0.0, float(raw))
    except ValueError:
        return float(default)


def _request_for_codex_account(request: CodexExecRequest, account: _CodexAccount) -> CodexExecRequest:
    codex_home = account.codex_home if account.codex_home is not None else request.codex_home
    return replace(
        request,
        codex_home=codex_home,
        codex_account_name=account.name,
        codex_top_level_args=(*account.top_level_args, *request.codex_top_level_args),
    )


def _codex_env_overrides(request: CodexExecRequest) -> dict[str, str]:
    if request.codex_home is None:
        return {}
    return {"CODEX_HOME": str(request.codex_home)}


def _append_codex_quota_event(request: CodexExecRequest, event: dict[str, object]) -> None:
    path = request.output_jsonl_path.with_suffix(request.output_jsonl_path.suffix + ".quota.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_unix": time.time(),
        "role": request.role,
        **event,
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _run_codex_exec_request_once(request: CodexExecRequest) -> CodexExecResult:
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
    run_env = build_local_execution_env(_codex_env_overrides(request))
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

    process: subprocess.Popen[str] | None = None
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        try:
            process = subprocess.Popen(
                command,
                cwd=request.cwd,
                env=run_env,
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
        except KeyboardInterrupt:
            if process is not None:
                _kill_process_tree(process)
                try:
                    stdout, _ = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    _kill_process_tree(process)
                    stdout, _ = process.communicate()
                jsonl_file.write(stdout)
            _append_jsonl_error(jsonl_file, request.role, "interrupted", "Codex invocation interrupted by user")
            raise
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
        codex_account_name=request.codex_account_name,
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
        codex_home=None if request.codex_home is None else Path(request.codex_home).expanduser().resolve(),
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
    codex_home = request.codex_home or Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).expanduser()
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
    if "401 unauthorized" in combined:
        return "codex_auth_failed", _first_nonempty(messages)
    if "input exceeds the maximum length" in combined:
        return "codex_input_too_large", _first_nonempty(messages)
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
