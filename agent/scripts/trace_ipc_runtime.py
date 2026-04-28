from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.ir_schema.actions import StepActionIR
from agent.ir_schema.program import normalize_ir, parse_ir_payload
from agent.runtime.action_dispatch import dispatch_action
from agent.runtime.models import RuntimeState
from agent.runtime.setup import (
    build_runtime_context,
    configure_headless_if_needed,
    create_runtime_context,
    ensure_genesis_initialized,
)


def _to_builtin(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        try:
            return _to_builtin(value.model_dump(mode="python"))
        except Exception:
            pass
    if hasattr(value, "detach"):
        try:
            value = value.detach()
        except Exception:
            pass
    if hasattr(value, "cpu"):
        try:
            value = value.cpu()
        except Exception:
            pass
    if hasattr(value, "numpy"):
        try:
            value = value.numpy()
        except Exception:
            pass
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "tolist"):
        try:
            return _to_builtin(value.tolist())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "__dict__"):
        return {str(key): _to_builtin(val) for key, val in vars(value).items()}
    return value


def _emit(record: dict[str, Any], *, out_file) -> None:
    line = json.dumps(_to_builtin(record), ensure_ascii=False)
    print(line)
    if out_file is not None:
        out_file.write(line + "\n")
        out_file.flush()


def _full_qd_array(qd_value):
    from genesis.utils.misc import qd_to_torch

    tensor = qd_to_torch(qd_value, transpose=True, copy=True)
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return np.asarray(tensor)


def _vec3_sum_from_links(qd_value, link_indices: list[int], env_idx: int) -> list[float] | None:
    if not link_indices:
        return None
    data = _full_qd_array(qd_value)
    if data.ndim == 2:
        selected = data[link_indices]
    elif data.ndim == 3:
        selected = data[env_idx, link_indices]
    else:
        return None
    total = np.sum(np.asarray(selected, dtype=np.float64), axis=0)
    return [float(total[0]), float(total[1]), float(total[2])]


def _rotation_angle_from_matrices(current: np.ndarray, target: np.ndarray) -> float:
    rel = current @ target.T
    trace = float(np.trace(rel))
    cosine = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    return float(math.acos(cosine))


def _entity_state(entity) -> dict[str, Any]:
    return {
        "pos": _to_builtin(entity.get_pos()),
        "quat": _to_builtin(entity.get_quat()),
        "vel": _to_builtin(entity.get_vel()),
        "ang": _to_builtin(entity.get_ang()),
    }


def _entity_link_indices(entity) -> list[int]:
    links = getattr(entity, "links", None)
    if links:
        return [int(link.idx) for link in links]
    base_link = getattr(entity, "base_link", None)
    if base_link is not None:
        return [int(base_link.idx)]
    return []


def _entity_link_objects(entity) -> list[Any]:
    links = getattr(entity, "links", None)
    if links:
        return list(links)
    base_link = getattr(entity, "base_link", None)
    if base_link is not None:
        return [base_link]
    return []


def _entity_name_for_geom(geom_idx: int, entities: dict[str, Any]) -> str | None:
    for entity_name, entity in entities.items():
        geom_start = getattr(entity, "geom_start", None)
        geom_end = getattr(entity, "geom_end", None)
        if geom_start is None or geom_end is None:
            continue
        if int(geom_start) <= geom_idx < int(geom_end):
            return entity_name
    return None


def _rigid_contact_entries(entity, entities: dict[str, Any]) -> list[dict[str, Any]]:
    get_contacts = getattr(entity, "get_contacts", None)
    if get_contacts is None:
        return []
    try:
        contact_data = get_contacts()
    except Exception:
        return []

    if not isinstance(contact_data, dict):
        return []

    geom_a = np.asarray(_to_builtin(contact_data.get("geom_a", [])))
    geom_b = np.asarray(_to_builtin(contact_data.get("geom_b", [])))
    pos = np.asarray(_to_builtin(contact_data.get("position", [])))
    force_a = np.asarray(_to_builtin(contact_data.get("force_a", [])))
    force_b = np.asarray(_to_builtin(contact_data.get("force_b", [])))
    valid_mask = np.asarray(_to_builtin(contact_data.get("valid_mask", []))) if "valid_mask" in contact_data else None

    if geom_a.ndim == 0:
        geom_a = geom_a.reshape(1)
    if geom_b.ndim == 0:
        geom_b = geom_b.reshape(1)
    if pos.ndim == 1 and pos.size:
        pos = pos.reshape(1, -1)
    if force_a.ndim == 1 and force_a.size:
        force_a = force_a.reshape(1, -1)
    if force_b.ndim == 1 and force_b.size:
        force_b = force_b.reshape(1, -1)
    if valid_mask is not None and valid_mask.ndim == 0:
        valid_mask = valid_mask.reshape(1)

    target_start = int(entity.geom_start)
    target_end = int(entity.geom_end)
    entries: list[dict[str, Any]] = []
    count = min(len(geom_a), len(geom_b))
    for index in range(count):
        if valid_mask is not None and index < len(valid_mask) and not bool(valid_mask[index]):
            continue
        geom_a_i = int(geom_a[index])
        geom_b_i = int(geom_b[index])
        a_is_target = target_start <= geom_a_i < target_end
        b_is_target = target_start <= geom_b_i < target_end
        if a_is_target == b_is_target:
            continue
        other_geom = geom_b_i if a_is_target else geom_a_i
        other_entity = _entity_name_for_geom(other_geom, entities)
        if other_entity is None:
            other_entity = f"geom_{other_geom}"
        force_on_target = force_a[index] if a_is_target else force_b[index]
        position = pos[index] if len(pos) > index else None
        entries.append(
            {
                "other_entity": other_entity,
                "geom_a": geom_a_i,
                "geom_b": geom_b_i,
                "position": _to_builtin(position),
                "force_on_target": _to_builtin(force_on_target),
                "force_norm": float(np.linalg.norm(force_on_target)),
            }
        )
    return entries


def _rigid_contact_summary(entity, entities: dict[str, Any]) -> dict[str, Any]:
    entries = _rigid_contact_entries(entity, entities)
    grouped: dict[str, dict[str, Any]] = {}
    for entry in entries:
        bucket = grouped.setdefault(
            entry["other_entity"],
            {"count": 0, "force_sum": np.zeros(3, dtype=np.float64), "force_norm_sum": 0.0},
        )
        bucket["count"] += 1
        bucket["force_sum"] += np.asarray(entry["force_on_target"], dtype=np.float64)
        bucket["force_norm_sum"] += float(entry["force_norm"])
    grouped_out = {
        name: {
            "count": item["count"],
            "force_sum": _to_builtin(item["force_sum"]),
            "force_norm_sum": float(item["force_norm_sum"]),
        }
        for name, item in grouped.items()
    }
    return {"count": len(entries), "by_other_entity": grouped_out, "entries": entries}


def _link_contact_force_summary(entity) -> dict[str, Any] | None:
    get_force = getattr(entity, "get_links_net_contact_force", None)
    if get_force is None:
        return None
    try:
        force = get_force()
    except Exception:
        return None
    force_np = np.asarray(_to_builtin(force), dtype=np.float64)
    if force_np.ndim == 1:
        force_np = force_np.reshape(1, -1)
    total = np.sum(force_np, axis=0)
    return {
        "per_link": _to_builtin(force_np),
        "sum": [float(total[0]), float(total[1]), float(total[2])],
    }


def _ipc_entity_summary(coupler, entity, env_idx: int) -> dict[str, Any] | None:
    data = getattr(coupler, "_coupling_data", None)
    if data is None:
        return None

    link_objects = _entity_link_objects(entity)
    link_entries: list[dict[str, Any]] = []
    total_force = np.zeros(3, dtype=np.float64)
    total_torque = np.zeros(3, dtype=np.float64)
    for link in link_objects:
        idx_local = data.link_to_idx_local.get(link)
        if idx_local is None:
            continue
        ipc_transform = np.asarray(data.ipc_transforms[env_idx, idx_local], dtype=np.float64)
        aim_transform = np.asarray(data.aim_transforms[env_idx, idx_local], dtype=np.float64)
        out_force = np.asarray(data.out_forces[env_idx, idx_local], dtype=np.float64)
        out_torque = np.asarray(data.out_torques[env_idx, idx_local], dtype=np.float64)
        delta_pos = ipc_transform[:3, 3] - aim_transform[:3, 3]
        delta_angle = _rotation_angle_from_matrices(ipc_transform[:3, :3], aim_transform[:3, :3])
        total_force += out_force
        total_torque += out_torque
        link_entries.append(
            {
                "link_name": getattr(link, "name", f"link_{link.idx}"),
                "link_idx": int(link.idx),
                "link_mass": float(link.inertial_mass),
                "ipc_pos": _to_builtin(ipc_transform[:3, 3]),
                "aim_pos": _to_builtin(aim_transform[:3, 3]),
                "delta_pos": _to_builtin(delta_pos),
                "delta_pos_norm": float(np.linalg.norm(delta_pos)),
                "delta_angle_rad": float(delta_angle),
                "out_force": _to_builtin(out_force),
                "out_torque": _to_builtin(out_torque),
            }
        )

    if not link_entries:
        return None

    return {
        "link_count": len(link_entries),
        "total_out_force": _to_builtin(total_force),
        "total_out_torque": _to_builtin(total_torque),
        "links": link_entries,
    }


def _rigid_internal_force_summary(scene, entity, env_idx: int) -> dict[str, Any]:
    link_indices = _entity_link_indices(entity)
    links_state = scene.sim.rigid_solver.links_state
    applied_force = _vec3_sum_from_links(links_state.cfrc_applied_vel, link_indices, env_idx)
    applied_torque = _vec3_sum_from_links(links_state.cfrc_applied_ang, link_indices, env_idx)
    coupling_force = _vec3_sum_from_links(links_state.cfrc_coupling_vel, link_indices, env_idx)
    coupling_torque = _vec3_sum_from_links(links_state.cfrc_coupling_ang, link_indices, env_idx)
    return {
        "applied_force_world_est": None if applied_force is None else [-x for x in applied_force],
        "applied_torque_world_est": None if applied_torque is None else [-x for x in applied_torque],
        "coupling_force_world_est": None if coupling_force is None else [-x for x in coupling_force],
        "coupling_torque_world_est": None if coupling_torque is None else [-x for x in coupling_torque],
    }


def _trace_record(
    *,
    scene,
    entities: dict[str, Any],
    tracked_entity_name: str,
    action_index: int | None,
    action_op: str | None,
    step: int,
    global_substep: int,
    local_substep: int,
    env_idx: int,
    phase: str,
) -> dict[str, Any]:
    tracked_entity = entities[tracked_entity_name]
    coupler = scene.sim.coupler
    return {
        "type": "trace",
        "phase": phase,
        "action_index": action_index,
        "action_op": action_op,
        "step": step,
        "time_sec": float(scene.cur_t),
        "global_substep": global_substep,
        "local_substep": local_substep,
        "entity": tracked_entity_name,
        "entity_state": _entity_state(tracked_entity),
        "rigid_internal_forces": _rigid_internal_force_summary(scene, tracked_entity, env_idx),
        "ipc_coupling": _ipc_entity_summary(coupler, tracked_entity, env_idx),
        "rigid_contact_summary": _rigid_contact_summary(tracked_entity, entities),
        "rigid_net_contact_force": _link_contact_force_summary(tracked_entity),
    }


def _run_scene_step(scene, *, trace_substeps: bool, on_trace) -> None:
    sim = scene.sim
    sim.process_input(in_backward=False)
    for _ in range(sim.substeps):
        current_local_substep = int(sim.cur_substep_local)
        sim.substep(current_local_substep)
        sim._cur_substep_global += 1
        if trace_substeps:
            on_trace(current_local_substep)
    if sim.rigid_solver.is_active:
        sim.rigid_solver.clear_external_force()
    sim._sensor_manager.step()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trace IPC runtime configuration and per-step entity dynamics.")
    parser.add_argument("--ir", type=Path, required=True, help="Input IR JSON path.")
    parser.add_argument("--entity", type=str, required=True, help="Tracked rigid entity name.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSONL output path.")
    parser.add_argument("--start-step", type=int, default=0, help="First scene step to emit traces for.")
    parser.add_argument("--end-step", type=int, default=None, help="Last scene step to emit traces for.")
    parser.add_argument("--every", type=int, default=1, help="Emit one trace every N scene steps.")
    parser.add_argument(
        "--trace-substeps",
        action="store_true",
        help="Emit an additional trace after every internal simulator substep instead of only per scene step.",
    )
    parser.add_argument("--env-idx", type=int, default=0, help="Environment index to inspect for batched scenes.")
    parser.add_argument("--keep-render", action="store_true", help="Keep render config from IR.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable quaternion normalization.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional hard cap on executed scene steps.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    payload = json.loads(args.ir.read_text(encoding="utf-8"))
    program = parse_ir_payload(payload)
    if not args.no_normalize:
        program = normalize_ir(program)
    if not args.keep_render:
        program = program.model_copy(deep=True)
        program.scene.render = None

    configure_headless_if_needed(program)

    import genesis as gs

    out_file = None
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_file = args.out.open("w", encoding="utf-8")

    ensure_genesis_initialized(gs, program)
    runtime = create_runtime_context(gs, program)
    state = RuntimeState()
    current_stage = "create_runtime"
    current_action_index: int | None = None
    current_action_op: str | None = None
    try:
        sim = runtime.scene.sim
        coupler = sim.coupler
        is_ipc_coupler = type(coupler).__name__ == "IPCCoupler"
        config_record: dict[str, Any] = {
            "type": "config",
            "ir_path": str(args.ir),
            "tracked_entity": args.entity,
            "stage": "prebuild",
            "scene_backend": program.scene.backend,
            "sim": {
                "dt": float(sim.dt),
                "substeps": int(sim.substeps),
                "substep_dt": float(sim.dt / sim.substeps),
                "gravity": _to_builtin(sim.gravity),
            },
            "fem_options": _to_builtin(runtime.scene.fem_options.model_dump(mode="python")),
            "coupler_type": type(coupler).__name__,
        }
        if is_ipc_coupler:
            from genesis.engine.couplers.ipc_coupler.utils import build_ipc_scene_config

            config_record["coupler_options"] = _to_builtin(coupler.options.model_dump(mode="python"))
            config_record["ipc_scene_config"] = _to_builtin(build_ipc_scene_config(coupler.options, sim.options))
            config_record["ipc_constraint_strength_scaled"] = {
                "translation": float(coupler._constraint_strength_translation_scaled),
                "rotation": float(coupler._constraint_strength_rotation_scaled),
            }
        _emit(config_record, out_file=out_file)

        current_stage = "scene_build"
        runtime.scene.build()
        current_stage = "build_runtime_context"
        build_runtime_context(program, runtime, state)

        if args.entity not in runtime.entities:
            raise ValueError(f"Unknown entity `{args.entity}`. Available: {sorted(runtime.entities)}")

        total_steps_to_execute = 0
        for action in program.actions:
            if isinstance(action, StepActionIR):
                total_steps_to_execute += int(action.steps)
        _emit({"type": "summary", "planned_scene_steps": total_steps_to_execute}, out_file=out_file)

        executed_steps = 0
        for action_index, action in enumerate(program.actions):
            action_op = getattr(action, "op", type(action).__name__)
            current_stage = "dispatch_action"
            current_action_index = action_index
            current_action_op = action_op
            _emit(
                {
                    "type": "action",
                    "action_index": action_index,
                    "action_op": action_op,
                    "payload": _to_builtin(action.model_dump(mode="python")),
                },
                out_file=out_file,
            )

            if isinstance(action, StepActionIR):
                for _ in range(int(action.steps)):
                    if args.max_steps is not None and executed_steps >= args.max_steps:
                        break

                    def _emit_trace(local_substep_completed: int) -> None:
                        if not args.trace_substeps:
                            return
                        if state.sim_step < args.start_step:
                            return
                        if args.end_step is not None and state.sim_step > args.end_step:
                            return
                        if state.sim_step % args.every != 0:
                            return
                        _emit(
                            _trace_record(
                                scene=runtime.scene,
                                entities=runtime.entities,
                                tracked_entity_name=args.entity,
                                action_index=action_index,
                                action_op=action_op,
                                step=state.sim_step,
                                global_substep=int(runtime.scene.sim.cur_substep_global),
                                local_substep=int(local_substep_completed),
                                env_idx=args.env_idx,
                                phase="post_substep",
                            ),
                            out_file=out_file,
                        )

                    _run_scene_step(runtime.scene, trace_substeps=args.trace_substeps, on_trace=_emit_trace)
                    state.sim_step += 1
                    executed_steps += 1

                    if state.sim_step >= args.start_step and (args.end_step is None or state.sim_step <= args.end_step):
                        if state.sim_step % args.every == 0:
                            _emit(
                                _trace_record(
                                    scene=runtime.scene,
                                    entities=runtime.entities,
                                    tracked_entity_name=args.entity,
                                    action_index=action_index,
                                    action_op=action_op,
                                    step=state.sim_step,
                                    global_substep=int(runtime.scene.sim.cur_substep_global),
                                    local_substep=int(runtime.scene.sim.cur_substep_local),
                                    env_idx=args.env_idx,
                                    phase="post_step",
                                ),
                                out_file=out_file,
                            )

                    if args.end_step is not None and state.sim_step >= args.end_step:
                        break
                if (args.max_steps is not None and executed_steps >= args.max_steps) or (
                    args.end_step is not None and state.sim_step >= args.end_step
                ):
                    break
            else:
                dispatch_action(action_index, action, runtime, state)

        _emit(
            {
                "type": "final",
                "executed_scene_steps": state.sim_step,
                "entity_state": _entity_state(runtime.entities[args.entity]),
            },
            out_file=out_file,
        )
    except Exception as exc:  # noqa: BLE001
        _emit(
            {
                "type": "error",
                "stage": current_stage,
                "action_index": current_action_index,
                "action_op": current_action_op,
                "sim_step": state.sim_step,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            },
            out_file=out_file,
        )
        raise
    finally:
        if out_file is not None:
            out_file.close()
        try:
            runtime.scene.destroy()
        except Exception:
            pass
        try:
            gs.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
