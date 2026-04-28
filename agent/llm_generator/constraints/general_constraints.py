from __future__ import annotations

from ...ir_schema.program import RigidIR, normalize_ir, parse_ir_payload
from .observation_policy import ALLOWED_OBSERVE_FIELDS, validate_observation_policy
from .payload_normalization import GeneralIRValidationError, extract_first_json_object, sanitize_payload
from .render_defaults import (
    DEFAULT_RENDER_VIDEO_PATH,
    apply_default_render_to_payload,
    default_render_config,
    ensure_program_has_render,
    synchronize_render_timing,
)


def parse_sanitize_validate(
    payload_or_program: dict[str, object] | RigidIR,
    *,
    normalize: bool = True,
) -> RigidIR:
    if isinstance(payload_or_program, RigidIR):
        program = payload_or_program
    else:
        sanitized = sanitize_payload(apply_default_render_to_payload(dict(payload_or_program)))
        program = parse_ir_payload(sanitized)

    validate_observation_policy(program)
    if normalize:
        program = normalize_ir(program)
    program = ensure_program_has_render(program)
    return synchronize_render_timing(program)


__all__ = [
    "ALLOWED_OBSERVE_FIELDS",
    "DEFAULT_RENDER_VIDEO_PATH",
    "GeneralIRValidationError",
    "extract_first_json_object",
    "default_render_config",
    "apply_default_render_to_payload",
    "ensure_program_has_render",
    "synchronize_render_timing",
    "parse_sanitize_validate",
    "validate_observation_policy",
]
