from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from code_agent.utils.state_cache import StateCacheError, verify_state_cache_manifest


def run_render_only_replay(
    *,
    scene: Any,
    actors: Any,
    render_state: dict[str, Any] | None,
    cache_manifest: str | Path,
) -> dict[str, Any]:
    manifest = verify_state_cache_manifest(
        cache_manifest,
        require_npz=True,
        require_complete_actor_state=True,
    )
    if render_state is None:
        return {
            "replay_only": True,
            "physics_cache_manifest": str(Path(cache_manifest)),
            "frames_replayed": 0,
            "rendered": False,
        }

    build = getattr(scene, "build", None)
    if callable(build):
        build()

    frames = manifest["frames"]
    for frame in frames:
        npz_path = Path(cache_manifest)
        if npz_path.is_dir():
            npz_path = npz_path / "manifest.json"
        npz_path = npz_path.parent / str(frame["npz"])
        apply_cached_state_npz(npz_path, actors, actor_contracts=manifest["actor_contracts"])
        capture = render_state.get("capture_frame")
        if callable(capture):
            capture(render_state, int(frame["step"]))

    render_state["replay_only"] = True
    render_state["physics_cache_manifest"] = str(Path(cache_manifest))
    render_state["physics_cache_frame_count"] = len(frames)
    return {
        "replay_only": True,
        "physics_cache_manifest": str(Path(cache_manifest)),
        "frames_replayed": len(frames),
        "rendered": True,
    }


def apply_cached_state_npz(
    npz_path: str | Path,
    actors: Any,
    *,
    actor_contracts: list[dict[str, Any]] | None = None,
) -> None:
    records = _actor_records(actors)
    by_name = {record["name"]: record["entity"] for record in records}
    with np.load(npz_path, allow_pickle=False) as data:
        names = [str(name) for name in data["actor_names"].tolist()]
        if actor_contracts is not None:
            _apply_contracted_states(data, actor_contracts=actor_contracts, by_name=by_name, npz_path=Path(npz_path))
            return
        positions = data["positions"] if "positions" in data.files else None
        quats = data["quats"] if "quats" in data.files else None
        for index, name in enumerate(names):
            entity = by_name.get(name)
            if entity is None:
                continue
            pos = None if positions is None else np.asarray(positions[index], dtype=float)
            quat = None if quats is None else np.asarray(quats[index], dtype=float)
            _apply_pose(entity, pos=pos, quat=quat)


def _apply_contracted_states(
    data: Any,
    *,
    actor_contracts: list[dict[str, Any]],
    by_name: dict[str, Any],
    npz_path: Path,
) -> None:
    for contract in actor_contracts:
        name = str(contract["name"])
        entity = by_name.get(name)
        if entity is None:
            raise StateCacheError(f"replay actor {name!r} is missing from rebuilt scene: {npz_path}")
        index = int(contract["index"])
        mode = str(contract["replay_mode"])
        if mode == "none":
            continue
        if mode == "qpos":
            key = f"actor_{index:03d}_qpos"
            _require_applied(_apply_qpos(entity, np.asarray(data[key])), name=name, mode=mode, npz_path=npz_path)
        elif mode == "dofs_position":
            key = f"actor_{index:03d}_dofs_position"
            _require_applied(
                _apply_dofs_position(entity, np.asarray(data[key])), name=name, mode=mode, npz_path=npz_path
            )
        elif mode == "state_pos":
            key = f"actor_{index:03d}_state_pos"
            _require_applied(
                _apply_state_position(entity, np.asarray(data[key])), name=name, mode=mode, npz_path=npz_path
            )
        elif mode == "pose":
            pos = np.asarray(data["positions"][index], dtype=float)
            quat = np.asarray(data["quats"][index], dtype=float)
            _require_applied(_apply_pose(entity, pos=pos, quat=quat), name=name, mode=mode, npz_path=npz_path)
        else:
            raise StateCacheError(f"unsupported replay mode {mode!r} for actor {name!r}: {npz_path}")


def _require_applied(applied: bool, *, name: str, mode: str, npz_path: Path) -> None:
    if not applied:
        raise StateCacheError(f"could not apply {mode} replay state for actor {name!r}: {npz_path}")


def _apply_qpos(entity: Any, value: np.ndarray) -> bool:
    method = getattr(entity, "set_qpos", None)
    if not callable(method):
        return False
    for kwargs in ({"zero_velocity": True}, {}):
        try:
            method(value, **kwargs)
            return True
        except (TypeError, ValueError):
            continue
        except Exception:  # noqa: BLE001
            return False
    return False


def _apply_dofs_position(entity: Any, value: np.ndarray) -> bool:
    method = getattr(entity, "set_dofs_position", None)
    if not callable(method):
        return False
    for kwargs in ({"zero_velocity": True}, {}):
        try:
            method(value, **kwargs)
            return True
        except (TypeError, ValueError):
            continue
        except Exception:  # noqa: BLE001
            return False
    return False


def _apply_state_position(entity: Any, value: np.ndarray) -> bool:
    method = getattr(entity, "set_position", None)
    if not callable(method):
        return False
    try:
        method(value)
        return True
    except Exception:  # noqa: BLE001
        return False


def _apply_pose(entity: Any, *, pos: np.ndarray | None, quat: np.ndarray | None) -> bool:
    pos_applied = pos is None or not np.all(np.isfinite(pos))
    quat_applied = quat is None or not np.all(np.isfinite(quat))
    if pos is not None and np.all(np.isfinite(pos)):
        for method_name in ("set_pos", "set_position"):
            method = getattr(entity, method_name, None)
            if callable(method):
                try:
                    method(pos)
                    pos_applied = True
                    break
                except Exception:  # noqa: BLE001
                    pass
    if quat is not None and np.all(np.isfinite(quat)):
        for method_name in ("set_quat", "set_rotation"):
            method = getattr(entity, method_name, None)
            if callable(method):
                try:
                    method(quat)
                    quat_applied = True
                    break
                except Exception:  # noqa: BLE001
                    pass
    return pos_applied and quat_applied


def _actor_records(actors: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_entities: set[int] = set()

    def visit(value: Any, key: str | None = None) -> None:
        if isinstance(value, dict) and value.get("entity") is not None:
            if id(value["entity"]) in seen_entities:
                return
            seen_entities.add(id(value["entity"]))
            name = str(value.get("name") or value.get("actor_name") or key or f"actor_{len(records):03d}")
            records.append({"name": name, "entity": value["entity"]})
            return
        if any(hasattr(value, name) for name in ("get_state", "set_position", "set_pos", "get_pos")):
            if id(value) in seen_entities:
                return
            seen_entities.add(id(value))
            records.append({"name": str(key or f"actor_{len(records):03d}"), "entity": value})
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
