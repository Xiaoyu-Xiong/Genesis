from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...configs import CONFIGS


class GeneralIRValidationError(RuntimeError):
    pass


def extract_first_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < 0 or end <= start:
            raise GeneralIRValidationError("Model output does not contain a JSON object.")
        snippet = stripped[start : end + 1]
        try:
            payload = json.loads(snippet)
        except json.JSONDecodeError as exc:
            raise GeneralIRValidationError(f"Model output JSON parse failed: {exc}") from exc

    if not isinstance(payload, dict):
        raise GeneralIRValidationError("Model output must be a JSON object.")
    return payload


def sanitize_payload(payload: dict[str, object]) -> dict[str, object]:
    normalized = dict(payload)
    bodies = normalized.get("bodies")
    if isinstance(bodies, list):
        sanitized_bodies: list[object] = []
        for body_any in bodies:
            if not isinstance(body_any, dict):
                sanitized_bodies.append(body_any)
                continue
            body = dict(body_any)
            shape = body.get("shape")
            if isinstance(shape, dict):
                shape = dict(shape)
                if shape.get("kind") == "mesh":
                    file_path = shape.get("file")
                    if isinstance(file_path, str) and file_path.strip():
                        shape["file"] = _rewrite_generated_runtime_mesh_path(file_path)
                body["shape"] = shape
            if body.get("simulation_kind") == "deformable":
                collision = body.get("collision")
                if isinstance(collision, dict):
                    collision = dict(collision)
                    collision.pop("coup_friction", None)
                    collision.pop("coup_restitution", None)
                    if CONFIGS.deformable.simulation_backend == "fem_ipc":
                        collision.pop("friction", None)
                    body["collision"] = collision
            sanitized_bodies.append(body)
        normalized["bodies"] = sanitized_bodies

    actions = normalized.get("actions")
    if not isinstance(actions, list):
        return normalized

    sanitized: list[object] = []
    for action_any in actions:
        if not isinstance(action_any, dict):
            sanitized.append(action_any)
            continue
        action = dict(action_any)
        op = action.get("op")
        if isinstance(op, str):
            action["op"] = op.strip().lower()
        if action.get("op") == "observe":
            fields = action.get("fields")
            if isinstance(fields, list):
                normalized_fields: list[str] = []
                for item in fields:
                    if not isinstance(item, str):
                        continue
                    field = item.strip().lower()
                    if field not in normalized_fields:
                        normalized_fields.append(field)
                action["fields"] = normalized_fields
        sanitized.append(action)

    normalized["actions"] = sanitized
    return normalized


def _rewrite_generated_runtime_mesh_path(file_path: str) -> str:
    path = Path(file_path)
    if path.name != "model.obj" or path.parent.name != "textured":
        return file_path
    candidate = path.parent.parent / "processed" / "repaired.obj"
    return candidate.as_posix() if candidate.exists() else file_path
