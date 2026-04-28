from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..ir_schema.scene import RenderIR


@dataclass
class RuntimeContext:
    scene: Any
    camera: Any | None
    render: RenderIR | None
    entities: dict[str, Any]
    body_entities: dict[str, Any]


@dataclass(frozen=True)
class ActuatorBinding:
    kind: str
    dofs_idx_local: tuple[int, ...]


@dataclass
class RuntimeState:
    events: list[dict[str, Any]] = field(default_factory=list)
    actuators_by_entity: dict[str, dict[str, ActuatorBinding]] = field(default_factory=dict)
    sim_step: int = 0
    rendered_frames: int = 0
    recording_started: bool = False
    recording_stopped: bool = False
