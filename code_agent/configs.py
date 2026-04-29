from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True, frozen=True)
class CodexConfigs:
    """Static defaults for Codex-backed planner, worker, and critic calls."""

    planner_model: str = "gpt-5.5"
    worker_model: str = "gpt-5.5"
    critic_model: str = "gpt-5.5"
    reasoning_effort: str = "xhigh"
    prompt_cache_retention: Literal["in_memory", "24h"] = "24h"
    planner_sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "read-only"
    critic_sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "read-only"
    worker_sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "workspace-write"
    ask_for_approval: Literal["untrusted", "on-request", "never"] = "never"


@dataclass(slots=True, frozen=True)
class HarnessConfigs:
    """Static harness limits that should usually not be left to generated code."""

    max_parallel_workers: int = 4
    max_repair_rounds: int = 12
    max_worker_attempts: int = 20
    execution_timeout_sec: float = 1000.0
    default_backend: str = "gpu"
    require_scope_check: bool = True


@dataclass(slots=True, frozen=True)
class RuntimeConfigs:
    """Genesis runtime defaults for generated simulations."""

    sim_dt: float = 0.01
    sim_substeps: int = 1
    render_every_n_steps: int = 3
    render_fps: int = 33
    render_res: tuple[int, int] = (640, 480)


@dataclass(slots=True, frozen=True)
class DeformableConfigs:
    """FEM + IPC deformable defaults for generated simulations."""

    friction: float = 0.3
    tet_resolution: int = 2
    genesis_precision: str = "32"
    fem_model: Literal["linear", "stable_neohookean", "linear_corotated"] = "stable_neohookean"
    fem_hydroelastic_modulus: float = 1e7
    fem_friction_mu: float = 0.3
    fem_contact_resistance: float | None = None
    fem_hessian_invariant: bool = False
    ipc_newton_max_iterations: int | None = None
    ipc_newton_min_iterations: int | None = 3
    ipc_newton_tolerance: float | None = 0.02
    ipc_newton_ccd_tolerance: float | None = 0.5
    ipc_newton_use_adaptive_tolerance: bool | None = True
    ipc_newton_translation_tolerance: float | None = 0.05
    ipc_newton_semi_implicit_enable: bool | None = None
    ipc_newton_semi_implicit_beta_tolerance: float | None = None
    ipc_n_linesearch_iterations: int | None = 16
    ipc_linesearch_report_energy: bool | None = None
    ipc_linear_system_solver: Literal["linear_pcg", "direct"] | None = None
    ipc_linear_system_tolerance: float | None = None
    ipc_contact_enable: bool | None = None
    ipc_contact_d_hat: float = 0.002
    ipc_contact_friction_enable: bool = True
    ipc_contact_resistance: float = 1.5e5
    ipc_contact_eps_velocity: float = 0.02
    ipc_contact_constitution: Literal["ipc", "isometric"] = "ipc"
    ipc_collision_detection_method: Literal["linear_bvh", "spatial_hash"] = "linear_bvh"
    ipc_cfl_enable: bool | None = True
    ipc_sanity_check_enable: bool | None = False
    ipc_constraint_strength_translation: float = 0.5
    ipc_constraint_strength_rotation: float = 0.2
    ipc_enable_rigid_ground_contact: bool = False
    ipc_enable_rigid_rigid_contact: bool = False
    ipc_two_way_coupling: bool = True
    ipc_enable_rigid_dofs_sync: bool = True
    ipc_free_base_driven_by_ipc: bool = True


@dataclass(slots=True, frozen=True)
class CriticConfigs:
    """Single-pass critic video sampling defaults."""

    sample_every_sec: float = 0.5
    max_frames: int = 24
    max_width: int = 640


@dataclass(slots=True, frozen=True)
class MeshyRequestConfigs:
    """Meshy asset request defaults."""

    mesh_format: Literal["obj", "glb", "stl"] = "obj"
    ai_model: Literal["latest", "meshy-6", "meshy-5"] = "meshy-5"
    art_style: Literal["realistic", "sculpture"] = "realistic"
    should_remesh: bool = True
    topology: Literal["triangle", "quad"] = "triangle"
    target_polycount: int | None = 5000
    symmetry_mode: Literal["off", "auto", "on"] = "auto"
    moderation: bool = False
    negative_prompt: str | None = None
    auto_size: bool = False
    origin_at: Literal["bottom", "center"] | None = None
    poll_interval_sec: float = 2.0
    max_wait_sec: float = 500.0
    timeout_sec: float = 200.0
    texture_enabled: bool = True
    texture_ai_model: Literal["latest", "meshy-6", "meshy-5"] | None = None
    texture_enable_pbr: bool = False
    texture_remove_lighting: bool = True


@dataclass(slots=True, frozen=True)
class MeshRepairConfigs:
    """Mesh repair and texture transfer defaults."""

    component_count_face_cap: int = 100000
    min_component_faces: int = 100
    max_repair_attempts: int = 4
    merge_vertices: bool = True
    merge_digits_vertex: int | None = 6
    fix_normals: bool = True
    process_validate: bool = True
    keep_largest_component: bool = True
    ftetwild_edge_length_fac: float = 0.05
    ftetwild_edge_length_abs: float | None = None
    ftetwild_optimize: bool = True
    ftetwild_simplify: bool = True
    ftetwild_epsilon: float = 1e-3
    ftetwild_stop_energy: float = 10.0
    ftetwild_coarsen: bool = False
    ftetwild_num_threads: int = 0
    ftetwild_num_opt_iter: int = 80
    ftetwild_quiet: bool = True
    ftetwild_disable_filtering: bool = False
    texture_transfer_max_resolution: int = 1024
    texture_transfer_chunk_size: int = 200000


@dataclass(slots=True, frozen=True)
class Configs:
    codex: CodexConfigs
    harness: HarnessConfigs
    runtime: RuntimeConfigs
    deformable: DeformableConfigs
    critic: CriticConfigs
    meshy_request: MeshyRequestConfigs
    mesh_repair: MeshRepairConfigs


CONFIGS = Configs(
    codex=CodexConfigs(),
    harness=HarnessConfigs(),
    runtime=RuntimeConfigs(),
    deformable=DeformableConfigs(),
    critic=CriticConfigs(),
    meshy_request=MeshyRequestConfigs(),
    mesh_repair=MeshRepairConfigs(),
)
