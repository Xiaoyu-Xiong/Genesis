from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from code_agent.io_utils import dump_json, load_json_object


_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_OWNERS = {"scene", "body", "action", "xml"}
_SCALES = {"linear", "log"}
_GROUPS = {
    "timing",
    "target",
    "control",
    "actuator",
    "initial",
    "layout",
    "geometry",
    "material",
    "contact",
    "solver",
    "other",
}
_TRANSFORMS = {
    "identity",
    "absolute_error",
    "squared_error",
    "threshold_min",
    "threshold_max",
    "reward_if_true",
    "penalty_if_true",
}
_OPS = {"<", "<=", ">", ">=", "==", "!="}


class OptContractError(ValueError):
    """Raised when an optimization contract cannot be consumed safely."""


@dataclass(slots=True, frozen=True)
class OptVariable:
    name: str
    default: float
    lower: float
    upper: float
    scale: str
    owner: str
    description: str
    group: str | None = None
    units: str | None = None
    active: bool = True
    initial_sigma: float | None = None

    @property
    def span(self) -> float:
        return self.upper - self.lower


@dataclass(slots=True, frozen=True)
class OptContracts:
    case_dir: Path
    target_spec: dict[str, Any]
    opt_space: dict[str, Any]
    variables: tuple[OptVariable, ...]
    default_params: dict[str, Any]
    target_spec_path: Path
    opt_space_path: Path
    default_params_path: Path

    @property
    def active_variables(self) -> tuple[OptVariable, ...]:
        return tuple(variable for variable in self.variables if variable.active)


def load_opt_contracts(
    *,
    case_dir: Path,
    target_spec_path: Path | None = None,
    opt_space_path: Path | None = None,
    default_params_path: Path | None = None,
    write_missing_default_params: bool = True,
) -> OptContracts:
    """Load and validate the optimization contracts for one case workspace."""

    case_dir = case_dir.resolve()
    contracts_dir = case_dir / "contracts"
    target_spec_path = (target_spec_path or contracts_dir / "target_spec.json").resolve()
    opt_space_path = (opt_space_path or contracts_dir / "opt_space.json").resolve()
    default_params_path = (default_params_path or contracts_dir / "default_opt_params.json").resolve()

    target_spec = _load_required_json(target_spec_path)
    opt_space = _load_required_json(opt_space_path)
    _validate_target_spec(target_spec, target_spec_path)
    variables = _variables_from_space(opt_space, opt_space_path)
    generated_defaults = params_payload_from_variables(variables, source="default")
    loaded_defaults = load_json_object(default_params_path)
    if loaded_defaults is None:
        default_params = generated_defaults
        if write_missing_default_params:
            dump_json(default_params, default_params_path)
    else:
        _validate_params_payload(loaded_defaults, default_params_path)
        default_params = merge_params_payloads(generated_defaults, loaded_defaults, source="default")

    _validate_params_against_variables(default_params, variables, default_params_path)
    return OptContracts(
        case_dir=case_dir,
        target_spec=target_spec,
        opt_space=opt_space,
        variables=tuple(variables),
        default_params=default_params,
        target_spec_path=target_spec_path,
        opt_space_path=opt_space_path,
        default_params_path=default_params_path,
    )


def params_payload_from_variables(
    variables: tuple[OptVariable, ...] | list[OptVariable],
    *,
    source: str | None,
    trial_index: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for variable in variables:
        set_dotted(params, variable.name, variable.default)
    payload: dict[str, Any] = {"schema_version": 1, "source": source, "trial_index": trial_index, "params": params}
    if metadata:
        payload["metadata"] = dict(metadata)
    return payload


def merge_params_payloads(
    base_payload: dict[str, Any],
    override_payload: dict[str, Any],
    *,
    source: str | None = None,
    trial_index: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = copy.deepcopy(base_payload)
    merged["params"] = deep_merge_dicts(
        copy.deepcopy(base_payload.get("params", {})),
        copy.deepcopy(override_payload.get("params", {})),
    )
    merged["source"] = source if source is not None else override_payload.get("source")
    merged["trial_index"] = trial_index if trial_index is not None else override_payload.get("trial_index")
    merged_metadata = deep_merge_dicts(
        copy.deepcopy(base_payload.get("metadata", {})),
        copy.deepcopy(override_payload.get("metadata", {})),
    )
    if metadata:
        merged_metadata = deep_merge_dicts(merged_metadata, copy.deepcopy(metadata))
    if merged_metadata:
        merged["metadata"] = merged_metadata
    elif "metadata" in merged:
        del merged["metadata"]
    return merged


def payload_from_vector(
    variables: tuple[OptVariable, ...] | list[OptVariable],
    vector: list[float] | tuple[float, ...],
    *,
    base_payload: dict[str, Any] | None = None,
    source: str | None = "trial",
    trial_index: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_variables = [variable for variable in variables if variable.active]
    if len(vector) != len(active_variables):
        raise OptContractError(f"Candidate vector has {len(vector)} values for {len(active_variables)} variables.")
    payload = (
        copy.deepcopy(base_payload)
        if base_payload is not None
        else params_payload_from_variables(variables, source=source)
    )
    payload["schema_version"] = 1
    payload["source"] = source
    payload["trial_index"] = trial_index
    payload.setdefault("params", {})
    for variable, raw_value in zip(active_variables, vector):
        set_dotted(payload["params"], variable.name, decode_normalized_value(variable, float(raw_value)))
    if metadata:
        payload["metadata"] = deep_merge_dicts(copy.deepcopy(payload.get("metadata", {})), metadata)
    return payload


def vector_from_payload(
    variables: tuple[OptVariable, ...] | list[OptVariable],
    payload: dict[str, Any],
) -> list[float]:
    params = payload.get("params", {})
    return [encode_value(variable, get_dotted(params, variable.name)) for variable in variables if variable.active]


def write_params_payload(path: Path, payload: dict[str, Any]) -> None:
    dump_json(payload, path)


def encode_value(variable: OptVariable, value: Any) -> float:
    value_float = _finite_number(value, f"params.{variable.name}")
    if variable.scale == "log":
        encoded = (math.log(value_float) - math.log(variable.lower)) / (
            math.log(variable.upper) - math.log(variable.lower)
        )
    else:
        encoded = (value_float - variable.lower) / variable.span
    return min(1.0, max(0.0, float(encoded)))


def decode_normalized_value(variable: OptVariable, value: float) -> float:
    normalized = min(1.0, max(0.0, float(value)))
    if variable.scale == "log":
        log_value = math.log(variable.lower) + normalized * (math.log(variable.upper) - math.log(variable.lower))
        return float(math.exp(log_value))
    return float(variable.lower + normalized * variable.span)


def normalized_initial_sigma(variable: OptVariable) -> float | None:
    if variable.initial_sigma is None:
        return None
    if 0.0 < variable.initial_sigma <= 1.0:
        return float(variable.initial_sigma)
    if variable.scale == "linear":
        return min(1.0, max(1e-4, float(variable.initial_sigma) / variable.span))
    return None


def set_dotted(container: dict[str, Any], path: str, value: Any) -> None:
    current = container
    parts = path.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def get_dotted(container: Any, path: str, default: Any = None) -> Any:
    if isinstance(container, dict) and path in container:
        return container[path]
    current = container
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return default
            current = current[index]
        else:
            return default
    return current


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def _load_required_json(path: Path) -> dict[str, Any]:
    payload = load_json_object(path)
    if payload is None:
        raise OptContractError(f"Missing or invalid JSON contract: {path}")
    return payload


def _validate_target_spec(target_spec: dict[str, Any], path: Path) -> None:
    if target_spec.get("schema_version") != 1:
        raise OptContractError(f"{path} must have schema_version 1.")
    if not isinstance(target_spec.get("task_family"), str) or not target_spec["task_family"]:
        raise OptContractError(f"{path} must define non-empty task_family.")
    objective = target_spec.get("objective")
    if not isinstance(objective, dict):
        raise OptContractError(f"{path} must define objective.")
    if objective.get("type") != "weighted_terms":
        raise OptContractError("Version 1 opt agent supports objective.type == 'weighted_terms' only.")
    if objective.get("direction") not in {"maximize", "minimize"}:
        raise OptContractError(f"{path} objective.direction must be maximize or minimize.")
    terms = objective.get("terms")
    if not isinstance(terms, list) or not terms:
        raise OptContractError(f"{path} objective.terms must be a non-empty list.")
    for index, term in enumerate(terms):
        _validate_objective_term(term, path, index)
    criteria = target_spec.get("success_criteria", [])
    if criteria is None:
        return
    if not isinstance(criteria, list):
        raise OptContractError(f"{path} success_criteria must be a list when provided.")
    for index, criterion in enumerate(criteria):
        if not isinstance(criterion, dict):
            raise OptContractError(f"{path} success_criteria[{index}] must be an object.")
        if not isinstance(criterion.get("name"), str) or not criterion["name"]:
            raise OptContractError(f"{path} success_criteria[{index}].name must be non-empty.")
        if not isinstance(criterion.get("metric_path"), str) or not criterion["metric_path"]:
            raise OptContractError(f"{path} success_criteria[{index}].metric_path must be non-empty.")
        if criterion.get("op") not in _OPS:
            raise OptContractError(f"{path} success_criteria[{index}].op is unsupported.")
        if "threshold" not in criterion:
            raise OptContractError(f"{path} success_criteria[{index}] must define threshold.")


def _validate_objective_term(term: Any, path: Path, index: int) -> None:
    if not isinstance(term, dict):
        raise OptContractError(f"{path} objective.terms[{index}] must be an object.")
    name = term.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise OptContractError(f"{path} objective.terms[{index}].name is invalid.")
    if not isinstance(term.get("metric_path"), str) or not term["metric_path"]:
        raise OptContractError(f"{path} objective.terms[{index}].metric_path must be non-empty.")
    _finite_number(term.get("weight"), f"{path} objective.terms[{index}].weight")
    transform = term.get("transform")
    if transform not in _TRANSFORMS:
        raise OptContractError(f"{path} objective.terms[{index}].transform is unsupported.")
    if transform in {"absolute_error", "squared_error"} and "target" not in term:
        raise OptContractError(f"{path} objective.terms[{index}] transform {transform} requires target.")
    if transform in {"threshold_min", "threshold_max"} and "target" not in term and "success_threshold" not in term:
        raise OptContractError(f"{path} objective.terms[{index}] transform {transform} requires target or threshold.")


def _variables_from_space(opt_space: dict[str, Any], path: Path) -> list[OptVariable]:
    if opt_space.get("schema_version") != 1:
        raise OptContractError(f"{path} must have schema_version 1.")
    if opt_space.get("optimizer") != "cma_es":
        raise OptContractError(f"{path} optimizer must be cma_es for version 1.")
    budget = opt_space.get("budget")
    if not isinstance(budget, dict):
        raise OptContractError(f"{path} must define budget.")
    max_trials = budget.get("max_trials")
    if not isinstance(max_trials, int) or isinstance(max_trials, bool) or max_trials < 1:
        raise OptContractError(f"{path} budget.max_trials must be an integer >= 1.")
    population_size = budget.get("population_size")
    if population_size is not None and (
        not isinstance(population_size, int) or isinstance(population_size, bool) or population_size < 3
    ):
        raise OptContractError(f"{path} budget.population_size must be null or an integer >= 3.")
    best_repeat_trials = budget.get("best_repeat_trials")
    if best_repeat_trials is not None and (
        not isinstance(best_repeat_trials, int) or isinstance(best_repeat_trials, bool) or best_repeat_trials < 1
    ):
        raise OptContractError(f"{path} budget.best_repeat_trials must be an integer >= 1.")
    _validate_strategy(opt_space.get("strategy"), path)
    variables_payload = opt_space.get("variables")
    if not isinstance(variables_payload, list) or not variables_payload:
        raise OptContractError(f"{path} variables must be a non-empty list.")
    variables: list[OptVariable] = []
    names: set[str] = set()
    for index, payload in enumerate(variables_payload):
        variable = _variable_from_payload(payload, path, index)
        if variable.name in names:
            raise OptContractError(f"{path} variable {variable.name!r} is duplicated.")
        names.add(variable.name)
        variables.append(variable)
    if not any(variable.active for variable in variables):
        raise OptContractError(f"{path} must expose at least one active variable.")
    return variables


def _variable_from_payload(payload: Any, path: Path, index: int) -> OptVariable:
    if not isinstance(payload, dict):
        raise OptContractError(f"{path} variables[{index}] must be an object.")
    name = payload.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise OptContractError(f"{path} variables[{index}].name is invalid.")
    if payload.get("type") != "float":
        raise OptContractError(f"{path} variables[{index}].type must be float.")
    lower, upper = _bounds(payload.get("bounds"), f"{path} variables[{index}].bounds")
    default = _finite_number(payload.get("default"), f"{path} variables[{index}].default")
    if default < lower or default > upper:
        raise OptContractError(f"{path} variable {name!r} default is outside bounds.")
    scale = payload.get("scale")
    if scale not in _SCALES:
        raise OptContractError(f"{path} variable {name!r} has unsupported scale.")
    if scale == "log" and (lower <= 0.0 or upper <= 0.0 or default <= 0.0):
        raise OptContractError(f"{path} variable {name!r} uses log scale but has non-positive values.")
    owner = payload.get("owner")
    if owner not in _OWNERS:
        raise OptContractError(f"{path} variable {name!r} has unsupported owner.")
    description = payload.get("description")
    if not isinstance(description, str) or not description:
        raise OptContractError(f"{path} variable {name!r} must define description.")
    group = payload.get("group")
    if group is not None and group not in _GROUPS:
        raise OptContractError(f"{path} variable {name!r} has unsupported group.")
    active = payload.get("active", True)
    if not isinstance(active, bool):
        raise OptContractError(f"{path} variable {name!r} active must be boolean when provided.")
    initial_sigma = payload.get("initial_sigma")
    if initial_sigma is not None:
        initial_sigma = _finite_number(initial_sigma, f"{path} variable {name!r}.initial_sigma")
        if initial_sigma <= 0.0:
            raise OptContractError(f"{path} variable {name!r}.initial_sigma must be positive.")
    return OptVariable(
        name=name,
        default=default,
        lower=lower,
        upper=upper,
        scale=scale,
        owner=owner,
        description=description,
        group=group,
        units=payload.get("units"),
        active=active,
        initial_sigma=initial_sigma,
    )


def _validate_strategy(strategy: Any, path: Path) -> None:
    if strategy is None:
        return
    if not isinstance(strategy, dict):
        raise OptContractError(f"{path} strategy must be an object when provided.")
    early_stop = strategy.get("early_stop")
    if early_stop is not None and not isinstance(early_stop, dict):
        raise OptContractError(f"{path} strategy.early_stop must be an object.")
    if isinstance(early_stop, dict):
        _optional_positive_int(early_stop.get("patience_generations"), f"{path} strategy.early_stop.patience_generations")
        _optional_non_negative_number(early_stop.get("min_delta"), f"{path} strategy.early_stop.min_delta")
    _validate_restarts(strategy.get("restarts"), path, "strategy.restarts")
    phases = strategy.get("phases")
    if phases is None:
        return
    if not isinstance(phases, list):
        raise OptContractError(f"{path} strategy.phases must be a list.")
    for index, phase in enumerate(phases):
        if not isinstance(phase, dict):
            raise OptContractError(f"{path} strategy.phases[{index}] must be an object.")
        _optional_positive_int(phase.get("max_trials"), f"{path} strategy.phases[{index}].max_trials")
        _optional_positive_number(phase.get("sigma_scale"), f"{path} strategy.phases[{index}].sigma_scale")
        _optional_population(phase.get("population_size"), f"{path} strategy.phases[{index}].population_size")
        groups = phase.get("groups")
        if groups is not None:
            if not isinstance(groups, list) or not all(isinstance(item, str) for item in groups):
                raise OptContractError(f"{path} strategy.phases[{index}].groups must be a string list.")
        names = phase.get("variables")
        if names is not None:
            if not isinstance(names, list) or not all(isinstance(item, str) and _NAME_RE.match(item) for item in names):
                raise OptContractError(f"{path} strategy.phases[{index}].variables must contain valid variable names.")
        _validate_restarts(phase.get("restarts"), path, f"strategy.phases[{index}].restarts")


def _validate_restarts(restarts: Any, path: Path, label: str) -> None:
    if restarts is None:
        return
    if not isinstance(restarts, list):
        raise OptContractError(f"{path} {label} must be a list.")
    for index, restart in enumerate(restarts):
        if not isinstance(restart, dict):
            raise OptContractError(f"{path} {label}[{index}] must be an object.")
        _optional_positive_int(restart.get("max_trials"), f"{path} {label}[{index}].max_trials")
        _optional_positive_number(restart.get("sigma_scale"), f"{path} {label}[{index}].sigma_scale")
        _optional_population(restart.get("population_size"), f"{path} {label}[{index}].population_size")


def _validate_params_payload(payload: dict[str, Any], path: Path) -> None:
    if payload.get("schema_version") != 1:
        raise OptContractError(f"{path} must have schema_version 1.")
    if not isinstance(payload.get("params"), dict):
        raise OptContractError(f"{path} must define object field params.")
    source = payload.get("source")
    if source not in {"default", "current", "best", "trial", "manual", None}:
        raise OptContractError(f"{path} has unsupported source {source!r}.")
    trial_index = payload.get("trial_index")
    if trial_index is not None and (not isinstance(trial_index, int) or trial_index < 0):
        raise OptContractError(f"{path} trial_index must be null or integer >= 0.")


def _validate_params_against_variables(payload: dict[str, Any], variables: list[OptVariable], path: Path) -> None:
    params = payload.get("params", {})
    for variable in variables:
        value = get_dotted(params, variable.name)
        if value is None:
            raise OptContractError(f"{path} is missing params.{variable.name}.")
        value_float = _finite_number(value, f"{path} params.{variable.name}")
        if value_float < variable.lower or value_float > variable.upper:
            raise OptContractError(f"{path} params.{variable.name}={value_float} is outside bounds.")
        if variable.scale == "log" and value_float <= 0.0:
            raise OptContractError(f"{path} params.{variable.name} must be positive for log scale.")


def _bounds(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise OptContractError(f"{label} must contain exactly two finite numbers.")
    lower = _finite_number(value[0], f"{label}[0]")
    upper = _finite_number(value[1], f"{label}[1]")
    if upper <= lower:
        raise OptContractError(f"{label} upper bound must be greater than lower bound.")
    return lower, upper


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise OptContractError(f"{label} must be a finite number.")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise OptContractError(f"{label} must be finite.")
    return value_float


def _optional_positive_int(value: Any, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise OptContractError(f"{label} must be null or an integer >= 1.")


def _optional_population(value: Any, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < 3:
        raise OptContractError(f"{label} must be null or an integer >= 3.")


def _optional_positive_number(value: Any, label: str) -> None:
    if value is None:
        return
    value_float = _finite_number(value, label)
    if value_float <= 0.0:
        raise OptContractError(f"{label} must be positive.")


def _optional_non_negative_number(value: Any, label: str) -> None:
    if value is None:
        return
    value_float = _finite_number(value, label)
    if value_float < 0.0:
        raise OptContractError(f"{label} must be non-negative.")
