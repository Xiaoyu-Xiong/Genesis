from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from code_agent.io_utils import load_json_object
from code_agent.opt.contracts import OptContracts, OptVariable


PARALLEL_MODES = {
    "auto",
    "subprocess_serial",
    "subprocess_parallel",
}

ENV_LOCAL_GROUPS = {"control", "initial"}
SCENE_LOCAL_GROUPS = {"material", "contact", "solver", "layout", "geometry", "other"}
TOPOLOGY_TOKENS = (
    "asset",
    "file",
    "mesh",
    "path",
    "resolution",
    "segment",
    "tet",
    "topolog",
    "vertex",
    "voxel",
)


@dataclass(slots=True, frozen=True)
class OptParallelPolicy:
    mode: str = "auto"
    max_batch_size: int | None = None
    force_serial: bool = False
    subprocess_workers: int | None = None
    gpu_memory_limit_gb: float | None = None
    gpu_memory_reserve_gb: float = 2.0
    subprocess_gpu_increment_gb: float | None = None

    def to_report(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class VariableParallelProfile:
    variable_names: tuple[str, ...]
    has_scene_local: bool
    has_topology_changing: bool
    reasons: tuple[str, ...]

    @property
    def requires_scene_rebuild(self) -> bool:
        return self.has_scene_local or self.has_topology_changing


@dataclass(slots=True, frozen=True)
class TrialExecutionPlan:
    backend: str
    workers: int
    batch_size: int
    reason: str
    variable_profile: VariableParallelProfile
    memory_profile: MemoryIncrementProfile


@dataclass(slots=True, frozen=True)
class MemoryIncrementProfile:
    usable_gpu_memory_gb: float | None
    reserve_gb: float
    subprocess_increment_gb: float
    subprocess_capacity: int
    source: str


def resolve_parallel_policy(opt_space: dict[str, Any]) -> OptParallelPolicy:
    execution = opt_space.get("execution")
    raw_policy = execution.get("parallel_policy") if isinstance(execution, dict) else None
    policy = raw_policy if isinstance(raw_policy, dict) else {}
    mode = str(policy.get("mode") or "auto")
    if mode not in PARALLEL_MODES:
        mode = "auto"
    memory_policy = _memory_policy_payload(policy, execution)
    return OptParallelPolicy(
        mode=mode,
        max_batch_size=_optional_positive_int(policy.get("max_batch_size")),
        force_serial=bool(policy.get("force_serial", False)),
        subprocess_workers=_optional_positive_int(policy.get("subprocess_workers")),
        gpu_memory_limit_gb=_optional_positive_float(memory_policy.get("gpu_memory_limit_gb")),
        gpu_memory_reserve_gb=_optional_non_negative_float(memory_policy.get("gpu_memory_reserve_gb"), default=2.0),
        subprocess_gpu_increment_gb=_optional_positive_float(memory_policy.get("subprocess_gpu_increment_gb")),
    )


def plan_trial_execution(
    *,
    policy: OptParallelPolicy,
    contracts: OptContracts,
    contracts_dir: Path,
    case_dir: Path,
    request_variable_names: tuple[str, ...],
    request_count: int,
) -> TrialExecutionPlan:
    del case_dir
    profile = classify_variables(contracts, request_variable_names)
    high_memory = _looks_like_deformable_or_ipc_case(contracts=contracts, contracts_dir=contracts_dir)
    memory_profile = resolve_memory_increment_profile(
        policy=policy,
        contracts=contracts,
        contracts_dir=contracts_dir,
        high_memory=high_memory,
    )
    batch_size = _batch_size(policy, request_count, memory_profile.subprocess_capacity)

    if request_count <= 1:
        return _plan("subprocess_serial", 1, 1, "single_trial", profile, memory_profile)
    if policy.force_serial or policy.mode == "subprocess_serial":
        return _plan("subprocess_serial", 1, 1, "policy_subprocess_serial", profile, memory_profile)

    reason = "policy_subprocess_parallel" if policy.mode == "subprocess_parallel" else "auto_subprocess_capacity"
    return _subprocess_or_serial_plan(
        policy,
        request_count,
        batch_size,
        profile,
        memory_profile,
        reason,
    )


def classify_variables(contracts: OptContracts, variable_names: tuple[str, ...]) -> VariableParallelProfile:
    variables_by_name = {variable.name: variable for variable in contracts.active_variables}
    selected = tuple(variables_by_name[name] for name in variable_names if name in variables_by_name)
    if not selected:
        selected = contracts.active_variables

    reasons: list[str] = []
    has_scene_local = False
    has_topology_changing = False
    for variable in selected:
        scope = _variable_scope(variable)
        reasons.append(f"{variable.name}:{scope}")
        if scope in {"scene", "topology"}:
            has_scene_local = True
        if scope == "topology":
            has_topology_changing = True
    return VariableParallelProfile(
        variable_names=tuple(variable.name for variable in selected),
        has_scene_local=has_scene_local,
        has_topology_changing=has_topology_changing,
        reasons=tuple(reasons),
    )


def resolve_memory_increment_profile(
    *,
    policy: OptParallelPolicy,
    contracts: OptContracts,
    contracts_dir: Path,
    high_memory: bool,
) -> MemoryIncrementProfile:
    opt_memory = contracts.opt_space.get("memory_profile")
    execution = contracts.opt_space.get("execution")
    execution_memory = execution.get("memory_profile") if isinstance(execution, dict) else None
    contract_memory = opt_memory if isinstance(opt_memory, dict) else {}
    execution_memory = execution_memory if isinstance(execution_memory, dict) else {}
    stored_memory = load_json_object(contracts_dir / "opt_memory_profile.json")
    stored_memory = stored_memory if isinstance(stored_memory, dict) else {}
    memory = {**contract_memory, **execution_memory, **stored_memory}

    usable = (
        policy.gpu_memory_limit_gb
        or _optional_positive_float(memory.get("gpu_memory_limit_gb"))
        or _query_gpu_free_memory_gb()
    )
    reserve = policy.gpu_memory_reserve_gb
    if memory.get("gpu_memory_reserve_gb") is not None and policy.gpu_memory_reserve_gb == 2.0:
        reserve = _optional_non_negative_float(memory.get("gpu_memory_reserve_gb"), default=2.0)
    if usable is not None:
        usable = max(0.0, usable - reserve)

    subprocess_default = 8.5 if high_memory else 2.5
    subprocess_increment = (
        policy.subprocess_gpu_increment_gb
        or _optional_positive_float(memory.get("subprocess_gpu_increment_gb"))
        or subprocess_default
    )
    return MemoryIncrementProfile(
        usable_gpu_memory_gb=usable,
        reserve_gb=reserve,
        subprocess_increment_gb=subprocess_increment,
        subprocess_capacity=_capacity_without_shared_fixed(usable, subprocess_increment),
        source=_memory_profile_source(policy, memory, usable),
    )


def _variable_scope(variable: OptVariable) -> str:
    name = variable.name.lower()
    group = variable.group or ""
    if any(token in name for token in TOPOLOGY_TOKENS):
        return "topology"
    if variable.owner == "action" or group in ENV_LOCAL_GROUPS:
        return "env"
    if group in SCENE_LOCAL_GROUPS or variable.owner in {"scene", "body", "xml"}:
        return "scene"
    return "scene"


def _looks_like_deformable_or_ipc_case(*, contracts: OptContracts, contracts_dir: Path) -> bool:
    deformable_config = load_json_object(contracts_dir / "deformable_config.json")
    if isinstance(deformable_config, dict):
        if bool(deformable_config.get("enabled")) or bool(deformable_config.get("ipc_enabled")):
            return True
        if (
            "enabled" not in deformable_config
            and "ipc_enabled" not in deformable_config
            and any(str(key).startswith("ipc_") and value is not None for key, value in deformable_config.items())
        ):
            return True

    execution = contracts.opt_space.get("execution")
    if isinstance(execution, dict) and bool(execution.get("high_memory_simulation")):
        return True
    for variable in contracts.active_variables:
        name = variable.name.lower()
        if variable.group in {"material", "contact", "solver"} and any(
            token in name for token in ("fem", "ipc", "deformable", "young", "poisson", "tet")
        ):
            return True
    return False


def _subprocess_or_serial_plan(
    policy: OptParallelPolicy,
    request_count: int,
    batch_size: int,
    profile: VariableParallelProfile,
    memory_profile: MemoryIncrementProfile,
    reason: str,
) -> TrialExecutionPlan:
    if memory_profile.subprocess_capacity <= 1:
        return _plan("subprocess_serial", 1, 1, f"{reason}_serial_capacity", profile, memory_profile)
    return _plan(
        "subprocess_parallel",
        _subprocess_workers(policy, request_count, memory_profile.subprocess_capacity),
        batch_size,
        reason,
        profile,
        memory_profile,
    )


def _plan(
    backend: str,
    workers: int,
    batch_size: int,
    reason: str,
    profile: VariableParallelProfile,
    memory_profile: MemoryIncrementProfile,
) -> TrialExecutionPlan:
    return TrialExecutionPlan(
        backend=backend,
        workers=max(1, workers),
        batch_size=max(1, batch_size),
        reason=reason,
        variable_profile=profile,
        memory_profile=memory_profile,
    )


def _batch_size(policy: OptParallelPolicy, request_count: int, capacity: int) -> int:
    batch_size = min(request_count, max(1, capacity))
    if policy.max_batch_size is not None:
        batch_size = min(batch_size, max(1, policy.max_batch_size))
    return max(1, batch_size)


def _subprocess_workers(policy: OptParallelPolicy, request_count: int, capacity: int) -> int:
    if policy.subprocess_workers is not None:
        return min(request_count, max(1, capacity), max(1, policy.subprocess_workers))
    cpu_count = os.cpu_count() or 2
    return min(request_count, max(1, capacity), max(1, min(2, cpu_count)))


def _capacity_without_shared_fixed(usable_gb: float | None, increment_gb: float) -> int:
    if usable_gb is None:
        return 1
    return max(1, int(usable_gb // increment_gb))


def _query_gpu_free_memory_gb() -> float | None:
    nvidia_smi = shutil.which("nvidia-smi") or _existing_path("/usr/lib/wsl/lib/nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        output = subprocess.check_output(
            [nvidia_smi, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            env=_gpu_query_env(),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return None
    values = []
    for line in output.splitlines():
        try:
            values.append(float(line.strip()) / 1024.0)
        except ValueError:
            continue
    return max(values) if values else None


def _existing_path(path: str) -> str | None:
    return path if Path(path).exists() else None


def _gpu_query_env() -> dict[str, str]:
    env = os.environ.copy()
    path_candidates = ["/usr/lib/wsl/lib"]
    current_parts = [part for part in env.get("PATH", "").split(os.pathsep) if part]
    existing = [part for part in path_candidates if Path(part).exists()]
    env["PATH"] = os.pathsep.join([*existing, *[part for part in current_parts if part not in existing]])
    return env


def _memory_policy_payload(policy_payload: dict[str, Any], execution: Any) -> dict[str, Any]:
    execution = execution if isinstance(execution, dict) else {}
    execution_memory = execution.get("memory_profile")
    execution_memory = execution_memory if isinstance(execution_memory, dict) else {}
    policy_memory = policy_payload.get("memory_profile")
    policy_memory = policy_memory if isinstance(policy_memory, dict) else {}
    return {**execution_memory, **policy_memory, **policy_payload}


def _memory_profile_source(policy: OptParallelPolicy, memory: dict[str, Any], usable: float | None) -> str:
    if (
        policy.gpu_memory_limit_gb is not None
        or policy.subprocess_gpu_increment_gb is not None
        or policy.gpu_memory_reserve_gb != 2.0
    ):
        return "policy"
    if memory:
        return "contract"
    if usable is not None:
        return "nvidia_smi_defaults"
    return "defaults_no_gpu_query"


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return max(1, parsed)


def _optional_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0.0:
        return None
    return parsed


def _optional_non_negative_float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, parsed)
