from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True, frozen=True)
class RuntimeConfigs:
    sim_dt: float = 0.01
    render_every_n_steps: int = 3
    render_res: tuple[int, int] = (640, 480)


@dataclass(slots=True, frozen=True)
class DeformableConfigs:
    simulation_backend: Literal["pbd", "fem_ipc"] = "fem_ipc"
    friction: float = 0.3
    tet_resolution: int = 2
    genesis_precision: str = "32"

    # PBD backend hyperparameters
    particle_size: float = 0.005
    max_stretch_solver_iterations: int = 12
    max_bending_solver_iterations: int = 2
    max_volume_solver_iterations: int = 12
    max_density_solver_iterations: int = 2
    max_viscosity_solver_iterations: int = 2
    stretch_relaxation: float = 0.1
    bending_relaxation: float = 0.1
    volume_relaxation: float = 0.1
    lower_bound: tuple[float, float, float] = (-100.0, -100.0, 0.0)
    upper_bound: tuple[float, float, float] = (100.0, 100.0, 100.0)

    # FEM + IPC backend hyperparameters
    fem_model: Literal["linear", "stable_neohookean", "linear_corotated"] = "stable_neohookean"
    fem_hydroelastic_modulus: float = 1e7
    fem_friction_mu: float = 0.3
    fem_contact_resistance: float | None = None
    fem_hessian_invariant: bool = False
    ipc_contact_d_hat: float = 0.002
    ipc_contact_friction_enable: bool = True
    ipc_contact_resistance: float = 1.5e5
    ipc_contact_eps_velocity: float = 0.01
    ipc_contact_constitution: Literal["ipc", "isometric"] = "ipc"
    ipc_collision_detection_method: Literal["linear_bvh", "spatial_hash"] = "linear_bvh"
    ipc_constraint_strength_translation: float = 0.7
    ipc_constraint_strength_rotation: float = 0.3
    ipc_enable_rigid_ground_contact: bool = False
    ipc_enable_rigid_rigid_contact: bool = True
    ipc_two_way_coupling: bool = True
    ipc_enable_rigid_dofs_sync: bool = False
    ipc_free_base_driven_by_ipc: bool = False


@dataclass(slots=True, frozen=True)
class OptimizationConfigs:
    model: str = "gpt-5.4"
    critic_model: str = ""
    reasoning_effort: str = "xhigh"
    critic_reasoning_effort: str = ""
    critic_prompt_variant: str = "full"
    max_parallel: int = 10
    backend: str = "gpu"
    max_opt_rounds: int = 12
    max_attempts: int = 20
    xml_max_attempts: int = 6
    sample_every_sec: float = 0.5
    max_frames: int = 24
    max_width: int = 640
    timeout_sec: float = 1000.0


@dataclass(slots=True, frozen=True)
class MeshyRequestConfigs:
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


@dataclass(slots=True, frozen=True)
class Configs:
    runtime: RuntimeConfigs
    deformable: DeformableConfigs
    optimization: OptimizationConfigs
    meshy_request: MeshyRequestConfigs
    mesh_repair: MeshRepairConfigs


CONFIGS = Configs(
    runtime=RuntimeConfigs(),
    deformable=DeformableConfigs(),
    optimization=OptimizationConfigs(),
    meshy_request=MeshyRequestConfigs(),
    mesh_repair=MeshRepairConfigs(),
)
