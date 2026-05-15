from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_ACTUATOR_RESPONSE_STEPS = 80


def run_actuator_response_check(
    xml_path: Path, *, steps: int = DEFAULT_ACTUATOR_RESPONSE_STEPS
) -> dict[str, Any]:
    """Apply bounded actuator commands in MuJoCo and confirm at least one generalized coordinate responds."""

    report: dict[str, Any] = {
        "ok": False,
        "xml_path": str(xml_path.resolve()),
        "steps": int(steps),
        "errors": [],
        "warnings": [],
        "actuators": [],
        "initial_qpos": [],
        "final_qpos": [],
        "qpos_delta_norm": 0.0,
    }
    try:
        import mujoco
        import numpy as np
    except Exception as exc:
        report["errors"].append(f"Actuator response check imports failed: {type(exc).__name__}: {exc}")
        return report

    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
    except Exception as exc:
        report["errors"].append(f"MuJoCo model setup failed: {type(exc).__name__}: {exc}")
        return report

    report["nu"] = int(model.nu)
    report["nq"] = int(model.nq)
    if model.nu <= 0:
        report["errors"].append("Model has no actuators.")
        return report
    if model.nq <= 0:
        report["errors"].append("Model has no generalized coordinates to move.")
        return report

    ctrl = _upper_control_vector(model)
    report["command"] = [float(value) for value in ctrl]
    report["actuators"] = [_actuator_info(mujoco, model, index, ctrl[index]) for index in range(model.nu)]
    try:
        mujoco.mj_forward(model, data)
        initial_qpos = data.qpos.copy()
        data.ctrl[:] = ctrl
        for _ in range(max(1, int(steps))):
            mujoco.mj_step(model, data)
        final_qpos = data.qpos.copy()
    except Exception as exc:
        report["errors"].append(f"Actuator response step failed: {type(exc).__name__}: {exc}")
        return report

    delta = final_qpos - initial_qpos
    report["initial_qpos"] = [float(value) for value in initial_qpos]
    report["final_qpos"] = [float(value) for value in final_qpos]
    report["qpos_delta"] = [float(value) for value in delta]
    delta_norm = float(np.linalg.norm(delta))
    report["qpos_delta_norm"] = delta_norm
    if delta_norm <= 1e-6:
        report["errors"].append("Actuator commands did not produce measurable qpos motion.")
    report["ok"] = not report["errors"]
    return report


def _upper_control_vector(model) -> Any:
    import numpy as np

    ctrl = np.zeros(model.nu, dtype=float)
    for index in range(model.nu):
        limited = bool(model.actuator_ctrllimited[index])
        if limited:
            lo, hi = model.actuator_ctrlrange[index]
            ctrl[index] = hi if abs(hi) >= abs(lo) else lo
        else:
            ctrl[index] = 1.0
    return ctrl


def _actuator_info(mujoco, model, index: int, command: float) -> dict[str, Any]:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or f"actuator_{index}"
    limited = bool(model.actuator_ctrllimited[index])
    ctrlrange = [float(value) for value in model.actuator_ctrlrange[index]] if limited else None
    return {
        "index": index,
        "name": name,
        "ctrllimited": limited,
        "ctrlrange": ctrlrange,
        "command": float(command),
    }
