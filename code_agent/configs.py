from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(slots=True, frozen=True)
class CodexConfigs:
    """Static defaults for Codex-backed planner, worker, and critic calls."""

    planner_model: str = "gpt-5.5"
    worker_model: str = "gpt-5.5"
    critic_model: str = "gpt-5.5"
    opt_model: str = "gpt-5.5"
    reasoning_effort: str = "xhigh"
    service_tier: Literal["fast", "standard"] | None = "standard"
    planner_sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "read-only"
    critic_sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "read-only"
    worker_sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "workspace-write"
    opt_sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "workspace-write"
    hide_builtin_assets_from_agents: bool = True
    builtin_asset_denied_roots: tuple[str, ...] = ("genesis/assets",)
    ask_for_approval: Literal["untrusted", "on-request", "never"] = "never"
    planner_timeout_sec: float = 1500.0
    worker_timeout_sec: float = 1500.0
    critic_timeout_sec: float = 1500.0
    opt_timeout_sec: float = 3600.0


@dataclass(slots=True, frozen=True)
class HarnessConfigs:
    """Static harness limits that should usually not be left to generated code."""

    max_parallel_cases: int | None = None
    max_parallel_workers: int | None = None
    max_repair_rounds: int = 12
    execution_timeout_sec: float = 1000.0
    command_timeout_sec: float = 300.0
    default_backend: str = "gpu"


@dataclass(slots=True, frozen=True)
class OptConfigs:
    """Static defaults for Opt agent requests and low-level numerical optimization."""

    enabled: bool = True
    agent_backend: Literal["cpu", "gpu"] = "gpu"
    agent_timeout_sec: float = 2000.0
    agent_render_baseline: bool = True
    agent_render_best: bool = True
    runner_timeout_sec: float = 1500.0
    runner_render_best: bool = True
    runner_baseline_trials: int = 1
    runner_best_repeat_trials: int = 2
    runner_default_initial_sigma: float = 0.25
    runner_early_stop_enabled: bool = True
    runner_early_stop_patience_generations: int = 3
    runner_early_stop_min_delta: float = 1e-4
    runner_stop_on_success: bool = True
    runner_boundary_near_margin: float = 0.03
    runner_boundary_warn_fraction: float = 0.5
    runner_restart_seed_stride: int = 1009
    runner_main_file: str = "src/main.py"
    runner_trial_root: str = "artifacts/opt_trials"
    runner_best_out_dir: str = "artifacts/opt_best"
    runner_current_params_path: str = "contracts/current_opt_params.json"
    cma_es_population_base: int = 4
    cma_es_population_log_multiplier: float = 3.0
    cma_es_low_dim_threshold: int = 8
    cma_es_low_dim_min_population: int = 6
    cma_es_low_dim_max_population: int = 8


@dataclass(slots=True, frozen=True)
class RuntimeConfigs:
    """Genesis runtime defaults for generated simulations."""

    sim_dt: float = 0.01
    sim_substeps: int = 10
    render_every_n_steps: int = 4
    render_fps: int = 25
    render_res: tuple[int, int] = (640, 480)


@dataclass(slots=True, frozen=True)
class DeformableConfigs:
    """FEM deformable defaults for generated simulations."""

    enabled: bool = False
    friction: float = 0.3
    tet_resolution: int = 2
    genesis_precision: str = "32"
    fem_density_default: float = 1000.0
    fem_density_min: float = 300.0
    fem_density_max: float = 3000.0
    fem_youngs_modulus_default: float = 1e5
    fem_youngs_modulus_min: float = 1e4
    fem_youngs_modulus_max: float = 5e6
    fem_poisson_ratio_default: float = 0.35
    fem_poisson_ratio_min: float = 0.0
    fem_poisson_ratio_max: float = 0.45
    fem_model: Literal["linear", "stable_neohookean", "linear_corotated"] = "stable_neohookean"
    fem_hydroelastic_modulus: float = 1e7
    fem_contact_resistance: float | None = None
    fem_hessian_invariant: bool = False


@dataclass(slots=True, frozen=True)
class IPCConfigs:
    """IPC contact/coupling defaults for generated simulations."""

    enabled: bool = False
    ipc_newton_max_iterations: int | None = None
    ipc_newton_min_iterations: int | None = 2
    ipc_newton_tolerance: float | None = None
    ipc_newton_ccd_tolerance: float | None = None
    ipc_newton_use_adaptive_tolerance: bool | None = True
    ipc_newton_translation_tolerance: float | None = None
    ipc_newton_semi_implicit_enable: bool | None = None
    ipc_newton_semi_implicit_beta_tolerance: float | None = None
    ipc_n_linesearch_iterations: int | None = None
    ipc_linesearch_report_energy: bool | None = None
    ipc_linear_system_solver: Literal["fused_pcg", "linear_pcg"] | None = "fused_pcg"
    ipc_linear_system_tolerance: float | None = None
    ipc_contact_enable: bool | None = None
    ipc_contact_d_hat: float = 0.01
    ipc_contact_d_hat_adaptive: bool = True
    ipc_contact_friction_enable: bool = True
    ipc_contact_resistance: float = 3e6
    ipc_contact_eps_velocity: float = 0.01
    ipc_contact_constitution: Literal["ipc", "al-ipc"] = "ipc"
    ipc_collision_detection_method: Literal["info_stackless_bvh", "stackless_bvh", "linear_bvh"] | None = (
        "info_stackless_bvh"
    )
    ipc_cfl_enable: bool | None = True
    ipc_sanity_check_enable: bool | None = True
    ipc_constraint_strength_translation: float = 10
    ipc_constraint_strength_rotation: float = 10
    ipc_enable_rigid_ground_contact: bool = False
    ipc_enable_rigid_rigid_contact: bool = True
    ipc_two_way_coupling: bool = True
    ipc_enable_rigid_dofs_sync: bool = True
    ipc_free_base_driven_by_ipc: bool = True


@dataclass(slots=True, frozen=True)
class CriticConfigs:
    """Single-pass critic video sampling defaults."""

    sample_every_sec: float = 0.5
    max_frames: int = 30
    max_width: int = 640
    max_attempts: int = 2
    prompt_inline_json_chars: int = 50000
    prompt_inline_text_chars: int = 24000


@dataclass(slots=True, frozen=True)
class MeshyRequestConfigs:
    """Meshy asset request defaults."""

    prompt_max_chars: int = 800
    max_parallel_api_requests: int | None = None
    max_parallel_local_processing: int = 1
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
    max_wait_sec: float = 1200.0
    timeout_sec: float = 300.0
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
    tetgen_sanity_timeout_sec: float = 120.0
    genesis_fem_import_timeout_sec: float = 240.0
    texture_transfer_max_resolution: int = 1024
    texture_transfer_chunk_size: int = 200000


@dataclass(slots=True, frozen=True)
class XMLAssetConfigs:
    """Codex MJCF/XML asset generation and validation defaults."""

    max_generation_attempts: int = 3
    max_parallel_workers: int | None = None
    preview_res: tuple[int, int] = (512, 512)
    preview_distance_scale: float = 2.5


@dataclass(slots=True, frozen=True)
class Configs:
    codex: CodexConfigs
    harness: HarnessConfigs
    opt: OptConfigs
    runtime: RuntimeConfigs
    deformable: DeformableConfigs
    ipc: IPCConfigs
    critic: CriticConfigs
    meshy_request: MeshyRequestConfigs
    mesh_repair: MeshRepairConfigs
    xml_asset: XMLAssetConfigs


CONFIGS = Configs(
    codex=CodexConfigs(),
    harness=HarnessConfigs(),
    opt=OptConfigs(),
    runtime=RuntimeConfigs(),
    deformable=DeformableConfigs(),
    ipc=IPCConfigs(),
    critic=CriticConfigs(),
    meshy_request=MeshyRequestConfigs(),
    mesh_repair=MeshRepairConfigs(),
    xml_asset=XMLAssetConfigs(),
)


def deformable_config_dict(
    *,
    enabled: bool | None = None,
    deformable_enabled: bool | None = None,
    ipc_enabled: bool | None = None,
) -> dict[str, object]:
    """Return the effective FEM/IPC config exposed to Planner and generated code."""

    if deformable_enabled is not None and enabled is not None and bool(deformable_enabled) != bool(enabled):
        raise ValueError("enabled and deformable_enabled disagree")
    if deformable_enabled is None:
        deformable_enabled = enabled

    data = asdict(CONFIGS.deformable)
    effective_deformable_enabled = (
        CONFIGS.deformable.enabled if deformable_enabled is None else bool(deformable_enabled)
    )
    requested_ipc_enabled = CONFIGS.ipc.enabled if ipc_enabled is None else bool(ipc_enabled)
    effective_ipc_enabled = bool(effective_deformable_enabled or requested_ipc_enabled)

    data["enabled"] = effective_deformable_enabled
    data.update({key: value for key, value in asdict(CONFIGS.ipc).items() if key != "enabled"})
    data["ipc_enabled"] = effective_ipc_enabled
    return data
