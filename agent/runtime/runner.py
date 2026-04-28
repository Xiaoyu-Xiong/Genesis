from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..ir_schema.program import RigidIR, normalize_ir, parse_ir_payload
from .action_dispatch import dispatch_action
from .helpers import capture_entity_snapshot, finalize_recording
from .models import RuntimeState
from .setup import build_runtime_context, configure_headless_if_needed, create_runtime_context, ensure_genesis_initialized


def run_rigid_ir(
    program_or_payload: Mapping[str, Any] | RigidIR,
    *,
    normalize: bool = True,
) -> dict[str, Any]:
    program = parse_ir_payload(program_or_payload)
    if normalize:
        program = normalize_ir(program)

    configure_headless_if_needed(program)

    import genesis as gs

    ensure_genesis_initialized(gs, program)
    runtime = create_runtime_context(gs, program)
    state = RuntimeState()
    scene_built = False
    result: dict[str, Any] = {
        "ir_version": program.ir_version,
        "status": "ok",
        "final_step": state.sim_step,
        "events": state.events,
    }
    current_action_index: int | None = None
    current_action: Any | None = None

    try:
        runtime.scene.build()
        scene_built = True
        build_runtime_context(program, runtime, state)

        for action_index, action in enumerate(program.actions):
            current_action_index = action_index
            current_action = action
            dispatch_action(action_index, action, runtime, state)
        result["final_step"] = state.sim_step
    except Exception as exc:  # noqa: BLE001
        default_entity_name = program.bodies[0].name if program.bodies else None
        crash_entity_name = getattr(current_action, "entity", default_entity_name if scene_built else None)
        attempted_step = state.sim_step + 1 if getattr(current_action, "op", None) == "step" else state.sim_step
        crash_snapshot = None
        if scene_built and isinstance(crash_entity_name, str):
            entity = runtime.entities.get(crash_entity_name)
            if entity is not None:
                crash_snapshot = capture_entity_snapshot(entity)

        crash = {
            "stage": "scene_build" if not scene_built else "action_dispatch",
            "action_index": current_action_index,
            "action_op": getattr(current_action, "op", None),
            "entity": crash_entity_name,
            "step": state.sim_step,
            "attempted_step": attempted_step,
            "time_sec": state.sim_step * float(program.scene.sim.dt),
            "attempted_time_sec": attempted_step * float(program.scene.sim.dt),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "snapshot": crash_snapshot,
        }
        state.events.append(
            {
                "type": "crash",
                "action_index": current_action_index,
                "step": state.sim_step,
                "attempted_step": attempted_step,
                "entity": crash_entity_name,
                "op": getattr(current_action, "op", None),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "state": crash_snapshot,
            }
        )
        result["status"] = "crashed"
        result["final_step"] = state.sim_step
        result["crash"] = crash
    finally:
        if scene_built:
            if (
                runtime.camera is not None
                and runtime.render is not None
                and state.recording_started
                and not state.recording_stopped
            ):
                try:
                    finalize_recording(runtime.camera, runtime.render)
                    state.recording_stopped = True
                    result["render"] = {
                        "video_path": runtime.render.output_video,
                        "fps": runtime.render.fps,
                        "frames": state.rendered_frames,
                    }
                except Exception:
                    pass
            runtime.scene.destroy()
        # Tear down Genesis global state as well, not just the scene. In GPU mode this
        # helps avoid stale CUDA/Taichi contexts when a long-lived worker process runs
        # multiple optimize rounds.
        try:
            gs.destroy()
        except Exception:
            pass
    return result
