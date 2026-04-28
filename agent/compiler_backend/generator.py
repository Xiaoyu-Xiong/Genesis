from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..ir_schema.program import RigidIR, normalize_ir, parse_ir_payload
from .emit_actions import emit_action_loop
from .emit_actuators import emit_actuator_setup
from .emit_scene import emit_scene_setup
from .runtime_helpers_source import runtime_helpers_source


@dataclass(frozen=True)
class CompiledRigidArtifact:
    program: RigidIR
    source: str


def compile_rigid_ir_to_source(
    program_or_payload: dict[str, Any] | RigidIR,
    *,
    function_name: str = "run",
) -> CompiledRigidArtifact:
    program = normalize_ir(parse_ir_payload(program_or_payload))
    lines: list[str] = []

    def emit(level: int, text: str = "") -> None:
        lines.append((" " * 4 * level) + text if text else "")

    lines.extend(["import os", "", "import json", "import genesis as gs", "", ""])
    lines.extend(runtime_helpers_source())
    lines.extend(["", "", f"def {function_name}():"])

    scene_ctx = emit_scene_setup(emit, program)
    emit_actuator_setup(emit, program=program, body_vars=scene_ctx.body_vars)
    emit_action_loop(emit, program=program, render=scene_ctx.render, entity_vars=scene_ctx.entity_vars)

    if scene_ctx.render is not None:
        emit(1, f"_video_path = {scene_ctx.render.output_video!r}")
        emit(1, "_video_dir = os.path.dirname(_video_path)")
        emit(1, "if _video_dir:")
        emit(2, "os.makedirs(_video_dir, exist_ok=True)")
        emit(1, f"camera.stop_recording(save_to_filename=_video_path, fps={scene_ctx.render.fps})")
        emit(1)

    emit(1, "return {")
    emit(2, f"'ir_version': {program.ir_version!r},")
    emit(2, "'final_step': sim_step,")
    emit(2, "'events': events,")
    if scene_ctx.render is not None:
        emit(
            2,
            f"'render': {{'video_path': {scene_ctx.render.output_video!r}, "
            f"'fps': {scene_ctx.render.fps}, 'frames': rendered_frames}},",
        )
    emit(1, "}")
    lines.extend(["", "", "if __name__ == '__main__':", f"    print(json.dumps({function_name}(), indent=2))"])
    return CompiledRigidArtifact(program=program, source="\n".join(lines))


def compile_rigid_ir_to_file(
    program_or_payload: dict[str, Any] | RigidIR,
    output_path: str | Path,
) -> CompiledRigidArtifact:
    artifact = compile_rigid_ir_to_source(program_or_payload=program_or_payload)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(artifact.source, encoding="utf-8")
    return artifact
