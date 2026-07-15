from __future__ import annotations

from typing import Literal


CodexSandbox = Literal["read-only", "workspace-write", "danger-full-access"]
Backend = Literal["cpu", "gpu"]
PhysicsMode = Literal["rigid", "rigid_ipc", "fem_ipc"]

DEFAULT_CODEX_MODEL = "gpt-5.5"
DEFAULT_CODEX_SANDBOX: CodexSandbox = "workspace-write"
DEFAULT_CODEX_REASONING_EFFORT = "high"
DEFAULT_CODEX_SERVICE_TIER: Literal["fast", "standard"] | None = "standard"
DEFAULT_CODEX_TIMEOUT_SEC = 20000.0

DEFAULT_BACKEND: Backend = "gpu"
DEFAULT_MAX_PARALLEL_CASES: int | None = None
DEFAULT_EXECUTION_TIMEOUT_SEC = 3600.0

RUNTIME_DEFAULTS = {
    "rigid": {
        "sim_dt": 0.01,
        "sim_substeps": 10,
        "render_every_n_steps": 4,
        "render_fps": 25,
        "render_res": (640, 480),
    },
    "ipc": {
        "sim_dt": 0.01,
        "sim_substeps": 1,
        "render_every_n_steps": 4,
        "render_fps": 25,
        "render_res": (640, 480),
    },
}

DEFORMABLE_BASE = {
    "friction": 0.3,
    "tet_resolution": 2,
    "genesis_precision": "32",
    "fem_density_default": 1000.0,
    "fem_density_min": 300.0,
    "fem_density_max": 3000.0,
    "fem_youngs_modulus_default": 1e5,
    "fem_youngs_modulus_min": 1e4,
    "fem_youngs_modulus_max": 5e6,
    "fem_poisson_ratio_default": 0.35,
    "fem_poisson_ratio_min": 0.0,
    "fem_poisson_ratio_max": 0.45,
    "fem_model": "stable_neohookean",
    "fem_hydroelastic_modulus": 1e7,
    "fem_contact_resistance": None,
    "fem_hessian_invariant": False,
    "cloth_density_default": 200.0,
    "cloth_youngs_modulus_default": 1e4,
    "cloth_poisson_ratio_default": 0.48,
    "cloth_thickness_default": 0.001,
    "cloth_bending_stiffness_default": 3.0,
    "cloth_friction_mu_default": 0.5,
    "cloth_target_edge_length_default": 0.01,
}

IPC_BASE = {
    "ipc_contact_d_hat": 0.01,
    "ipc_contact_d_hat_adaptive": True,
    "ipc_contact_friction_enable": True,
    "ipc_contact_resistance": 1e7,
    "ipc_constraint_strength_translation": 30,
    "ipc_constraint_strength_rotation": 30,
    "ipc_contact_eps_velocity": 0.01,
    "ipc_collision_detection_method": "info_stackless_bvh",
    "ipc_linear_system_solver": "fused_pcg",
    "ipc_cfl_enable": True,
    "ipc_sanity_check_enable": True,
    "ipc_enable_rigid_ground_contact": False,
    "ipc_enable_rigid_rigid_contact": True,
    "ipc_two_way_coupling": True,
    "ipc_enable_rigid_dofs_sync": True,
    "ipc_free_base_driven_by_ipc": True,
}


def deformable_config_dict(*, physics_mode: PhysicsMode = "rigid", deformable_kind: str = "none") -> dict[str, object]:
    if physics_mode == "fem_ipc":
        deformable_enabled = True
        ipc_enabled = True
    elif physics_mode == "rigid_ipc":
        deformable_enabled = False
        ipc_enabled = True
    elif physics_mode == "rigid":
        deformable_enabled = False
        ipc_enabled = False
    else:
        raise ValueError(f"Unsupported physics_mode: {physics_mode!r}")

    if not deformable_enabled:
        deformable_kind = "none"

    return {
        **DEFORMABLE_BASE,
        **IPC_BASE,
        "enabled": deformable_enabled,
        "deformable_kind": deformable_kind,
        "ipc_enabled": ipc_enabled,
    }


def deformable_config_from_planner_output(planner_output: dict[str, object]) -> dict[str, object]:
    physics_plan = planner_output.get("physics_plan")
    if not isinstance(physics_plan, dict):
        return deformable_config_dict()
    mode = str(physics_plan.get("mode") or "rigid")
    if mode not in {"rigid", "rigid_ipc", "fem_ipc"}:
        mode = "rigid"
    deformable_kind = str(physics_plan.get("deformable_kind") or "none")
    return deformable_config_dict(physics_mode=mode, deformable_kind=deformable_kind)


def runtime_defaults_dict(*, ipc_enabled: bool) -> dict[str, object]:
    return dict(RUNTIME_DEFAULTS["ipc" if ipc_enabled else "rigid"])
