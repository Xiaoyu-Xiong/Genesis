from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .ftetwild_backend import repair_mesh_with_ftetwild
from .models import MeshRepairConfig, MeshRepairResult


def repair_mesh_for_simulation(
    mesh_path: Path,
    output_dir: Path,
    config: MeshRepairConfig,
    *,
    attempt_index: int = 1,
    strategy_name: str | None = None,
) -> MeshRepairResult:
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    suffix = mesh_path.suffix.lower()
    attempt_output = processed_dir / f"repaired_attempt_{attempt_index:02d}{suffix}"
    return repair_mesh_with_ftetwild(
        mesh_path,
        output_dir,
        config,
        output_mesh_path=attempt_output,
        attempt_index=attempt_index,
        strategy_name=strategy_name or f"ftetwild_attempt_{attempt_index:02d}",
    )


def build_ftetwild_retry_configs(config: MeshRepairConfig) -> list[tuple[str, MeshRepairConfig]]:
    max_attempts = max(1, int(config.max_repair_attempts))
    ladder: list[tuple[str, MeshRepairConfig]] = [
        ("ftetwild_base", config),
        (
            "ftetwild_robust_medium",
            replace(
                config,
                ftetwild_edge_length_fac=max(config.ftetwild_edge_length_fac, 0.08),
                ftetwild_epsilon=max(config.ftetwild_epsilon, 0.0025),
                ftetwild_coarsen=True,
                ftetwild_optimize=True,
                ftetwild_simplify=True,
                ftetwild_stop_energy=max(config.ftetwild_stop_energy, 12.0),
            ),
        ),
        (
            "ftetwild_robust_coarse",
            replace(
                config,
                ftetwild_edge_length_fac=max(config.ftetwild_edge_length_fac, 0.12),
                ftetwild_epsilon=max(config.ftetwild_epsilon, 0.005),
                ftetwild_coarsen=True,
                ftetwild_optimize=True,
                ftetwild_simplify=True,
                ftetwild_stop_energy=max(config.ftetwild_stop_energy, 20.0),
            ),
        ),
        (
            "ftetwild_last_resort",
            replace(
                config,
                ftetwild_edge_length_fac=max(config.ftetwild_edge_length_fac, 0.18),
                ftetwild_epsilon=max(config.ftetwild_epsilon, 0.01),
                ftetwild_coarsen=True,
                ftetwild_optimize=False,
                ftetwild_simplify=True,
                ftetwild_stop_energy=max(config.ftetwild_stop_energy, 30.0),
                ftetwild_num_opt_iter=min(config.ftetwild_num_opt_iter, 20),
            ),
        ),
    ]
    return ladder[:max_attempts]
