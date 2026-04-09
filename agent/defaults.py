from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True, frozen=True)
class RuntimeDefaults:
    sim_dt: float = 0.005
    render_every_n_steps: int = 6
    render_res: tuple[int, int] = (640, 480)


@dataclass(slots=True, frozen=True)
class DeformableDefaults:
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
    ipc_contact_d_hat: float = 0.004
    ipc_contact_friction_enable: bool = True
    ipc_contact_resistance: float = 1e8
    ipc_contact_eps_velocity: float = 0.01
    ipc_contact_constitution: Literal["ipc", "isometric"] = "ipc"
    ipc_collision_detection_method: Literal["linear_bvh", "spatial_hash"] = "linear_bvh"
    ipc_constraint_strength_translation: float = 50.0
    ipc_constraint_strength_rotation: float = 50.0
    ipc_enable_rigid_ground_contact: bool = False
    ipc_enable_rigid_rigid_contact: bool = False
    ipc_two_way_coupling: bool = False
    ipc_enable_rigid_dofs_sync: bool = False
    ipc_free_base_driven_by_ipc: bool = False


@dataclass(slots=True, frozen=True)
class OptimizationDefaults:
    model: str = "gpt-5.2"
    critic_model: str = ""
    reasoning_effort: str = "xhigh"
    critic_reasoning_effort: str = ""
    critic_prompt_variant: str = "full"
    max_parallel: int = 10
    backend: str = "gpu"
    max_opt_rounds: int = 8
    max_attempts: int = 12
    xml_max_attempts: int = 4
    sample_every_sec: float = 0.5
    max_frames: int = 24
    max_width: int = 640
    timeout_sec: float = 1000.0


@dataclass(slots=True, frozen=True)
class Defaults:
    runtime: RuntimeDefaults
    deformable: DeformableDefaults
    optimization: OptimizationDefaults


DEFAULTS = Defaults(
    runtime=RuntimeDefaults(),
    deformable=DeformableDefaults(),
    optimization=OptimizationDefaults(),
)
