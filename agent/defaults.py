from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class RuntimeDefaults:
    sim_dt: float = 0.0005
    render_every_n_steps: int = 20
    render_res: tuple[int, int] = (640, 480)


@dataclass(slots=True, frozen=True)
class DeformableDefaults:
    friction: float = 0.5
    particle_size: float = 0.01
    tet_resolution: int = 3
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
    genesis_precision: str = "32"


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
