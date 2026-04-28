from __future__ import annotations

from ...ir_schema.actions import ObserveActionIR, StepActionIR
from ...ir_schema.program import RigidIR
from .payload_normalization import GeneralIRValidationError

ALLOWED_OBSERVE_FIELDS = {
    "pos",
    "quat",
    "vel",
    "ang",
    "qpos",
    "dofs_position",
    "dofs_velocity",
    "bbox_min",
    "bbox_max",
    "bbox_size",
    "vertex_disp_mean",
    "vertex_disp_max",
}


def validate_observation_policy(program: RigidIR) -> None:
    for idx, action in enumerate(program.actions):
        if not isinstance(action, ObserveActionIR):
            continue
        bad_fields = [field for field in action.fields if field not in ALLOWED_OBSERVE_FIELDS]
        if bad_fields:
            raise GeneralIRValidationError(
                f"Action[{idx}] observe fields {bad_fields} are invalid; "
                f"allowed: {sorted(ALLOWED_OBSERVE_FIELDS)}."
            )

    observe_steps, final_step = _observation_schedule(program)
    required_count = _required_observation_count(final_step)
    if len(observe_steps) < required_count:
        raise GeneralIRValidationError(
            "Observation policy not satisfied: "
            f"final_step={final_step} requires >= {required_count} observe actions, "
            f"but got {len(observe_steps)}."
        )
    if final_step > 0 and (not observe_steps or observe_steps[-1] != final_step):
        raise GeneralIRValidationError(
            "Observation policy not satisfied: final observation must be at the final simulation step."
        )
    if required_count >= 2 and not any(step < final_step for step in observe_steps):
        raise GeneralIRValidationError(
            "Observation policy not satisfied: simulations with steps require at least one pre-final observation."
        )
    if required_count >= 3 and not any(0 < step < final_step for step in observe_steps):
        raise GeneralIRValidationError(
            "Observation policy not satisfied: long simulations require at least one intermediate observation."
        )


def _required_observation_count(final_step: int) -> int:
    if final_step <= 0:
        return 1
    if final_step < 100:
        return 2
    return 3


def _observation_schedule(program: RigidIR) -> tuple[list[int], int]:
    sim_step = 0
    observe_steps: list[int] = []
    for action in program.actions:
        if isinstance(action, StepActionIR):
            sim_step += action.steps
        elif isinstance(action, ObserveActionIR):
            observe_steps.append(sim_step)
    return observe_steps, sim_step
