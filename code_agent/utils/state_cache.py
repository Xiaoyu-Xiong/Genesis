from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


STATE_CACHE_SCHEMA_VERSION = 2


class StateCacheError(RuntimeError):
    """Raised when a state cache is missing required replay artifacts."""


@dataclass
class StateCacheWriter:
    out_dir: Path
    scene: Any
    actors: Any
    steps: int
    sim_dt: float | None = None
    sim_substeps: int | None = None
    backend: str | None = None
    render_profile: str = "debug_raster"
    capture_steps: set[int] | None = None
    cache_dir: Path = field(init=False)
    states_dir: Path = field(init=False)
    frames: list[dict[str, Any]] = field(default_factory=list)
    _captured_steps: set[int] = field(default_factory=set)
    _actor_contracts: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.cache_dir = self.out_dir / "state_cache"
        self.states_dir = self.cache_dir / "states"
        self.states_dir.mkdir(parents=True, exist_ok=True)
        if self.capture_steps is not None:
            self.capture_steps = {int(step) for step in self.capture_steps}

    @classmethod
    def create(
        cls,
        *,
        out_dir: Path,
        scene: Any,
        actors: Any,
        steps: int,
        render_state: dict[str, Any] | None = None,
        sim_dt: float | None = None,
        sim_substeps: int | None = None,
        backend: str | None = None,
        render_profile: str = "debug_raster",
    ) -> "StateCacheWriter":
        capture_steps = None
        if isinstance(render_state, dict):
            raw_steps = render_state.get("capture_step_set") or render_state.get("capture_steps")
            if raw_steps is not None:
                capture_steps = {int(step) for step in raw_steps}
        return cls(
            out_dir=out_dir,
            scene=scene,
            actors=actors,
            steps=int(steps),
            sim_dt=sim_dt,
            sim_substeps=sim_substeps,
            backend=backend,
            render_profile=render_profile,
            capture_steps=capture_steps,
        )

    def should_capture(self, step: int) -> bool:
        step_i = int(step)
        if step_i in self._captured_steps:
            return False
        return self.capture_steps is None or step_i in self.capture_steps

    def capture(self, step: int) -> Path | None:
        step_i = int(step)
        if not self.should_capture(step_i):
            return None
        frame_index = len(self.frames)
        rel_path = Path("states") / f"frame_{frame_index:03d}.npz"
        npz_path = self.cache_dir / rel_path
        payload = self._state_payload(step_i)
        np.savez_compressed(npz_path, **payload)
        self._captured_steps.add(step_i)
        self.frames.append(
            {
                "index": frame_index,
                "step": step_i,
                "npz": str(rel_path),
                "actor_count": int(payload["actor_names"].shape[0]),
            }
        )
        return npz_path

    def make_capture_state(self) -> dict[str, Any]:
        return {
            "capture_step_set": set(self.capture_steps or range(int(self.steps) + 1)),
            "capture_steps": set(self.capture_steps or range(int(self.steps) + 1)),
            "capture_frame": lambda _state, step: self.capture(int(step)),
            "state_cache_writer": self,
        }

    def finalize(self, *, accepted_by_critic: bool | None = None) -> Path:
        manifest = {
            "schema_version": STATE_CACHE_SCHEMA_VERSION,
            "kind": "genesis_state_cache",
            "manifest": "manifest.json",
            "steps": int(self.steps),
            "sim_dt": self.sim_dt,
            "sim_substeps": self.sim_substeps,
            "backend": self.backend,
            "render_profile": self.render_profile,
            "accepted_by_critic": accepted_by_critic,
            "frame_count": len(self.frames),
            "frame_steps": [int(frame["step"]) for frame in self.frames],
            "frames": list(self.frames),
            "actor_names": [contract["name"] for contract in self._actor_contracts],
            "actor_contracts": list(self._actor_contracts),
            "state_completeness_required": True,
            "source_hashes": _source_hashes(self.out_dir),
            "npz_required": True,
        }
        manifest_path = self.cache_dir / "manifest.json"
        manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest_path

    def _state_payload(self, step: int) -> dict[str, Any]:
        records = _actor_records(self.actors)
        record_names = [str(record["name"]) for record in records]
        if len(set(record_names)) != len(record_names):
            raise StateCacheError(f"state cache actor names must be unique: {record_names}")
        expected_contract_count = len(self._actor_contracts)
        if expected_contract_count and len(records) != expected_contract_count:
            raise StateCacheError(
                f"actor count changed at step {step}: expected {expected_contract_count}, found {len(records)}"
            )
        names: list[str] = []
        positions: list[np.ndarray] = []
        quats: list[np.ndarray] = []
        arrays: dict[str, Any] = {
            "schema_version": np.asarray([STATE_CACHE_SCHEMA_VERSION], dtype=np.int32),
            "step": np.asarray([int(step)], dtype=np.int64),
        }
        for index, record in enumerate(records):
            names.append(record["name"])
            entity = record.get("entity")
            pos = _first_array(entity, ("get_pos", "get_position", "get_center_of_mass"))
            quat = _first_array(entity, ("get_quat", "get_rotation"))
            positions.append(_vector_or_nan(pos, 3))
            quats.append(_vector_or_nan(quat, 4))
            qpos = _method_array(entity, "get_qpos")
            dofs_position = _method_array(entity, "get_dofs_position")
            state_pos = _entity_state_pos(entity)
            if state_pos is not None and state_pos.size:
                arrays[f"actor_{index:03d}_state_pos"] = state_pos
            if qpos is not None and qpos.size:
                arrays[f"actor_{index:03d}_qpos"] = qpos
            if dofs_position is not None and dofs_position.size:
                arrays[f"actor_{index:03d}_dofs_position"] = dofs_position
            contract = _build_actor_contract(
                index=index,
                record=record,
                pos=positions[-1],
                quat=quats[-1],
                qpos=qpos,
                dofs_position=dofs_position,
                state_pos=state_pos,
            )
            if index < len(self._actor_contracts):
                _check_actor_contract_compatible(self._actor_contracts[index], contract, step=step)
            else:
                self._actor_contracts.append(contract)
        arrays["actor_names"] = np.asarray(names, dtype="U128")
        arrays["positions"] = np.asarray(positions, dtype=np.float64).reshape((len(names), 3))
        arrays["quats"] = np.asarray(quats, dtype=np.float64).reshape((len(names), 4))
        return arrays


def attach_state_cache_capture(render_state: dict[str, Any], writer: StateCacheWriter) -> dict[str, Any]:
    original = render_state.get("capture_frame") if isinstance(render_state, dict) else None

    def capture_with_cache(*args: Any, **kwargs: Any) -> Any:
        step = _capture_step_from_args(args, kwargs)
        writer.capture(step)
        if callable(original):
            return original(*args, **kwargs)
        return None

    render_state["capture_frame"] = capture_with_cache
    render_state["state_cache_writer"] = writer
    return render_state


def verify_state_cache_manifest(
    manifest_path: str | Path,
    *,
    require_npz: bool = True,
    require_complete_actor_state: bool = False,
) -> dict[str, Any]:
    path = Path(manifest_path)
    if path.is_dir():
        path = path / "manifest.json"
    if not path.is_file():
        raise StateCacheError(f"state cache manifest not found: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StateCacheError(f"state cache manifest is not valid JSON: {path}") from exc
    if not isinstance(manifest, dict):
        raise StateCacheError(f"state cache manifest must be an object: {path}")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise StateCacheError(f"state cache manifest has no frame entries: {path}")
    actor_contracts = _validated_actor_contracts(
        manifest,
        path=path,
        require_complete_actor_state=require_complete_actor_state,
    )
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise StateCacheError(f"state cache frame entry {index} is not an object: {path}")
        npz_value = frame.get("npz")
        if require_npz and not isinstance(npz_value, str):
            raise StateCacheError(f"state cache frame {index} is missing required npz path")
        if isinstance(npz_value, str):
            npz_path = path.parent / npz_value
            if not npz_path.is_file():
                raise StateCacheError(f"state cache frame {index} npz not found: {npz_path}")
            try:
                with np.load(npz_path, allow_pickle=False) as data:
                    if "step" not in data.files or "actor_names" not in data.files:
                        raise StateCacheError(f"state cache frame {index} npz lacks required arrays: {npz_path}")
                    npz_step = int(np.asarray(data["step"]).reshape(-1)[0])
                    if npz_step != int(frame.get("step", -1)):
                        raise StateCacheError(
                            f"state cache frame {index} step mismatch: manifest={frame.get('step')}, "
                            f"npz={npz_step}: {npz_path}"
                        )
                    if require_complete_actor_state:
                        _verify_frame_actor_state(
                            data,
                            actor_contracts=actor_contracts,
                            frame_index=index,
                            npz_path=npz_path,
                        )
            except StateCacheError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise StateCacheError(f"state cache frame {index} npz cannot be opened: {npz_path}") from exc
    return manifest


def _validated_actor_contracts(
    manifest: dict[str, Any],
    *,
    path: Path,
    require_complete_actor_state: bool,
) -> list[dict[str, Any]]:
    raw = manifest.get("actor_contracts")
    if not require_complete_actor_state:
        return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
    if int(manifest.get("schema_version") or 0) < 2 or manifest.get("state_completeness_required") is not True:
        raise StateCacheError(f"state cache lacks required complete actor-state contract: {path}")
    if not isinstance(raw, list) or not raw:
        raise StateCacheError(f"state cache has no actor state contracts: {path}")
    contracts: list[dict[str, Any]] = []
    for index, contract in enumerate(raw):
        if not isinstance(contract, dict):
            raise StateCacheError(f"state cache actor contract {index} is not an object: {path}")
        if int(contract.get("index", -1)) != index or not isinstance(contract.get("name"), str):
            raise StateCacheError(f"state cache actor contract {index} has invalid identity: {path}")
        replay_mode = str(contract.get("replay_mode") or "")
        required_arrays = contract.get("required_arrays")
        if replay_mode not in {"none", "pose", "qpos", "dofs_position", "state_pos"}:
            raise StateCacheError(f"state cache actor contract {index} has invalid replay mode: {replay_mode!r}")
        if not isinstance(required_arrays, list) or not all(isinstance(key, str) for key in required_arrays):
            raise StateCacheError(f"state cache actor contract {index} has invalid required_arrays: {path}")
        if replay_mode != "none" and not required_arrays:
            raise StateCacheError(f"state cache actor contract {index} has no required replay arrays: {path}")
        contracts.append(contract)
    manifest_names = manifest.get("actor_names")
    contract_names = [str(contract["name"]) for contract in contracts]
    if manifest_names != contract_names:
        raise StateCacheError(f"state cache actor_names do not match actor contracts: {path}")
    return contracts


def _verify_frame_actor_state(
    data: Any,
    *,
    actor_contracts: list[dict[str, Any]],
    frame_index: int,
    npz_path: Path,
) -> None:
    names = [str(name) for name in np.asarray(data["actor_names"]).tolist()]
    expected_names = [str(contract["name"]) for contract in actor_contracts]
    if names != expected_names:
        raise StateCacheError(f"state cache frame {frame_index} actor names do not match manifest contract: {npz_path}")
    for contract in actor_contracts:
        actor_index = int(contract["index"])
        expected_shapes = contract.get("array_shapes") if isinstance(contract.get("array_shapes"), dict) else {}
        for key in contract["required_arrays"]:
            if key not in data.files:
                raise StateCacheError(f"state cache frame {frame_index} lacks required actor array {key!r}: {npz_path}")
            value = np.asarray(data[key])
            if key in {"positions", "quats"}:
                if value.ndim < 2 or actor_index >= value.shape[0]:
                    raise StateCacheError(f"state cache frame {frame_index} has invalid {key!r} actor row: {npz_path}")
                value = value[actor_index]
            expected_shape = expected_shapes.get(key)
            if isinstance(expected_shape, list) and list(value.shape) != expected_shape:
                raise StateCacheError(
                    f"state cache frame {frame_index} actor array {key!r} changed shape "
                    f"from {expected_shape} to {list(value.shape)}: {npz_path}"
                )
            if value.size == 0 or not np.all(np.isfinite(value)):
                raise StateCacheError(
                    f"state cache frame {frame_index} actor array {key!r} is empty or non-finite: {npz_path}"
                )


def _capture_step_from_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    if "step" in kwargs:
        return int(kwargs["step"])
    if len(args) == 1:
        return int(args[0])
    if len(args) >= 2:
        return int(args[1])
    raise TypeError("state cache capture requires a step argument")


def _actor_records(actors: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_entities: set[int] = set()

    def visit(value: Any, key: str | None = None) -> None:
        entity = _extract_entity(value)
        if entity is not None:
            ident = id(entity)
            if ident in seen_entities:
                return
            seen_entities.add(ident)
            raw = value if isinstance(value, dict) else {}
            name = str(
                raw.get("name")
                or raw.get("actor_name")
                or raw.get("logical_name")
                or key
                or f"actor_{len(records):03d}"
            )
            records.append({"name": name, "entity": entity, "raw": raw})
            return
        if isinstance(value, dict):
            for child_key, child in value.items():
                if child_key in {"metadata", "meta", "path_samples"}:
                    continue
                visit(child, str(child_key))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"actor_{index:03d}")

    visit(actors)
    return records


def _extract_entity(value: Any) -> Any | None:
    if isinstance(value, dict):
        entity = value.get("entity")
        if entity is not None:
            return entity
    if hasattr(value, "get_state") or hasattr(value, "get_pos") or hasattr(value, "set_pos"):
        return value
    return None


def _first_array(entity: Any, method_names: tuple[str, ...]) -> np.ndarray | None:
    if entity is None:
        return None
    for method_name in method_names:
        method = getattr(entity, method_name, None)
        if not callable(method):
            continue
        try:
            return _to_numpy(method())
        except Exception:  # noqa: BLE001
            continue
    for attr_name in ("pos", "position"):
        if hasattr(entity, attr_name):
            try:
                return _to_numpy(getattr(entity, attr_name))
            except Exception:  # noqa: BLE001
                continue
    return None


def _entity_state_pos(entity: Any) -> np.ndarray | None:
    method = getattr(entity, "get_state", None)
    if not callable(method):
        return None
    try:
        state = method()
    except Exception:  # noqa: BLE001
        return None
    if isinstance(state, dict):
        raw = state.get("pos")
    else:
        raw = getattr(state, "pos", None)
    if raw is None:
        return None
    arr = _to_numpy(raw)
    if arr is None:
        return None
    return np.asarray(arr, dtype=np.float64)


def _method_array(entity: Any, method_name: str) -> np.ndarray | None:
    if entity is None:
        return None
    method = getattr(entity, method_name, None)
    if not callable(method):
        return None
    try:
        value = _to_numpy(method())
    except Exception:  # noqa: BLE001
        return None
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def _build_actor_contract(
    *,
    index: int,
    record: dict[str, Any],
    pos: np.ndarray,
    quat: np.ndarray,
    qpos: np.ndarray | None,
    dofs_position: np.ndarray | None,
    state_pos: np.ndarray | None,
) -> dict[str, Any]:
    entity = record.get("entity")
    qpos_key = f"actor_{index:03d}_qpos"
    dofs_key = f"actor_{index:03d}_dofs_position"
    state_pos_key = f"actor_{index:03d}_state_pos"
    available_arrays: list[str] = []
    array_shapes: dict[str, list[int]] = {}

    if qpos is not None and qpos.size:
        available_arrays.append(qpos_key)
        array_shapes[qpos_key] = list(qpos.shape)
    if dofs_position is not None and dofs_position.size:
        available_arrays.append(dofs_key)
        array_shapes[dofs_key] = list(dofs_position.shape)
    if state_pos is not None and state_pos.size:
        available_arrays.append(state_pos_key)
        array_shapes[state_pos_key] = list(state_pos.shape)
    if np.all(np.isfinite(pos)):
        available_arrays.append("positions")
        array_shapes["positions"] = [3]
    if np.all(np.isfinite(quat)):
        available_arrays.append("quats")
        array_shapes["quats"] = [4]

    n_dofs = _nonnegative_int_attr(entity, "n_dofs")
    n_links = _nonnegative_int_attr(entity, "n_links")
    has_qpos_setter = callable(getattr(entity, "set_qpos", None))
    has_dofs_setter = callable(getattr(entity, "set_dofs_position", None))
    has_state_pos_setter = callable(getattr(entity, "set_position", None))
    has_pose_setter = any(callable(getattr(entity, name, None)) for name in ("set_pos", "set_position"))
    state_pos_is_geometry = bool(
        state_pos is not None
        and state_pos.size > 3
        and state_pos.ndim >= 2
        and state_pos.shape[-1] == 3
        and has_state_pos_setter
    )

    if qpos_key in available_arrays and has_qpos_setter:
        replay_mode = "qpos"
        required_arrays = [key for key in (qpos_key, dofs_key) if key in available_arrays]
    elif dofs_key in available_arrays and has_dofs_setter:
        replay_mode = "dofs_position"
        required_arrays = [key for key in (qpos_key, dofs_key) if key in available_arrays]
    elif state_pos_is_geometry:
        replay_mode = "state_pos"
        required_arrays = [state_pos_key]
    elif has_pose_setter and {"positions", "quats"}.issubset(available_arrays):
        replay_mode = "pose"
        required_arrays = ["positions", "quats"]
    else:
        replay_mode = "none"
        required_arrays = []

    if n_dofs is not None and n_dofs > 0 and replay_mode not in {"qpos", "dofs_position"}:
        raise StateCacheError(f"actor {record['name']!r} has {n_dofs} DOFs but no replayable qpos/DOF-position state")

    raw_text = " ".join(
        f"{key}={value}".lower()
        for key, value in record.get("raw", {}).items()
        if key != "entity" and isinstance(value, (str, int, float, bool))
    )
    if n_dofs is not None and n_dofs > 0 and ((n_links or 0) > 1 or "articulat" in raw_text):
        state_kind = "articulated"
    elif replay_mode in {"qpos", "dofs_position", "pose"}:
        state_kind = "rigid"
    elif replay_mode == "state_pos":
        state_kind = "deformable_or_particles"
    else:
        state_kind = "static_or_metadata_only"

    return {
        "index": int(index),
        "name": str(record["name"]),
        "state_kind": state_kind,
        "n_dofs": n_dofs,
        "n_links": n_links,
        "replay_mode": replay_mode,
        "required_arrays": required_arrays,
        "available_arrays": available_arrays,
        "array_shapes": {key: array_shapes[key] for key in required_arrays},
    }


def _check_actor_contract_compatible(
    expected: dict[str, Any],
    current: dict[str, Any],
    *,
    step: int,
) -> None:
    stable_fields = ("index", "name", "state_kind", "replay_mode", "required_arrays", "array_shapes")
    changed = [field for field in stable_fields if expected.get(field) != current.get(field)]
    if changed:
        raise StateCacheError(
            f"actor state contract changed at step {step} for {expected.get('name')!r}: {', '.join(changed)}"
        )


def _nonnegative_int_attr(entity: Any, attr_name: str) -> int | None:
    try:
        value = int(getattr(entity, attr_name))
    except (AttributeError, TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    elif hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value)


def _vector_or_nan(value: np.ndarray | None, size: int) -> np.ndarray:
    out = np.full((size,), np.nan, dtype=np.float64)
    if value is None:
        return out
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    n = min(size, flat.size)
    if n:
        out[:n] = flat[:n]
    return out


def _source_hashes(out_dir: Path) -> dict[str, str]:
    case_root = Path(out_dir).resolve().parent
    src_dir = case_root / "src"
    hashes: dict[str, str] = {}
    if not src_dir.is_dir():
        return hashes
    for path in sorted(src_dir.glob("*.py")):
        try:
            rel = path.relative_to(case_root)
            hashes[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
    return hashes


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
