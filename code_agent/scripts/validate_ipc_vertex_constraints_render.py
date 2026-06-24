"""Render validation scenes for IPC FEM/cloth vertex constraints.

This is intentionally a runnable validation script rather than a pytest fixture:
it creates short real videos that are useful for manual inspection while also
checking basic numeric constraint error and non-blank frame statistics.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import genesis as gs
from genesis.utils.misc import tensor_to_array


STIFFNESS = 1.0e5
MAX_ALLOWED_ERROR = 0.12
MIN_NORMALIZED_FRAME_STD = 0.01
FEM_BAR_SIZE = (0.28, 0.08, 0.08)
CLOTH_WIDTH = 0.48
FEM_POS = (-0.60, 0.0, 0.34)
CLOTH_POS_LOW = (0.60, 0.0, 0.42)
CLOTH_POS_HIGH = (0.60, 0.0, 0.44)
CLAMP_SIZE = (0.065, 0.12, 0.12)
CLAMP_CLEARANCE = 0.008


@dataclass
class ConstraintProbe:
    name: str
    entity: Any
    verts_idx: np.ndarray
    target_fn: Callable[[], np.ndarray]


def _write_cloth_grid_obj(path: Path, *, nx: int = 9, ny: int = 7, width: float = 0.48, height: float = 0.32) -> Path:
    lines: list[str] = []
    for iy in range(ny):
        y = -0.5 * height + height * iy / (ny - 1)
        for ix in range(nx):
            x = -0.5 * width + width * ix / (nx - 1)
            lines.append(f"v {x:.8f} {y:.8f} 0.00000000")

    for iy in range(ny - 1):
        for ix in range(nx - 1):
            a = iy * nx + ix + 1
            b = a + 1
            c = a + nx
            d = c + 1
            lines.append(f"f {a} {b} {d}")
            lines.append(f"f {a} {d} {c}")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _frame_std_normalized(rgb_array: Any) -> float:
    arr = tensor_to_array(rgb_array).astype(np.float32)
    if arr.size == 0:
        return 0.0
    if arr.max(initial=0.0) > 2.0:
        arr /= 255.0
    return float(arr.reshape((-1, arr.shape[-1])).std(axis=0).max())


def _entity_positions(entity: Any) -> np.ndarray:
    return tensor_to_array(entity.get_state().pos)[0]


def _side_indices(entity: Any, axis: int, side: str, *, fraction: float = 0.18) -> np.ndarray:
    positions = _entity_positions(entity)
    values = positions[:, axis]
    lo = float(values.min())
    hi = float(values.max())
    width = max(hi - lo, 1.0e-8)
    if side == "min":
        selected = np.flatnonzero(values <= lo + fraction * width)
    elif side == "max":
        selected = np.flatnonzero(values >= hi - fraction * width)
    else:
        raise ValueError(side)
    if selected.size == 0:
        raise RuntimeError(f"no vertices selected for axis={axis} side={side}")
    return selected.astype(np.int32)


def _add_floor(scene: gs.Scene) -> None:
    scene.add_entity(
        morph=gs.morphs.Plane(pos=(0.0, 0.0, -0.05)),
        material=gs.materials.Rigid(needs_coup=False),
        surface=gs.surfaces.Plastic(color=(0.78, 0.82, 0.86, 1.0), roughness=0.9),
    )


def _add_fem_bar(scene: gs.Scene, *, pos: tuple[float, float, float]) -> Any:
    return scene.add_entity(
        morph=gs.morphs.Box(size=FEM_BAR_SIZE, pos=pos, tet_resolution=2),
        material=gs.materials.FEM.Elastic(E=2.0e4, nu=0.35, rho=300.0, model="stable_neohookean"),
        surface=gs.surfaces.Plastic(color=(0.92, 0.28, 0.18, 1.0), roughness=0.65),
    )


def _add_cloth_sheet(scene: gs.Scene, *, mesh_path: Path, pos: tuple[float, float, float]) -> Any:
    return scene.add_entity(
        morph=gs.morphs.Mesh(file=str(mesh_path), pos=pos),
        material=gs.materials.FEM.Cloth(E=1.5e4, nu=0.3, rho=80.0, thickness=0.002),
        surface=gs.surfaces.Plastic(color=(0.18, 0.42, 0.88, 1.0), roughness=0.7, double_sided=True),
    )


def _add_clamp(scene: gs.Scene, *, pos: tuple[float, float, float], color: tuple[float, float, float, float]) -> Any:
    return scene.add_entity(
        morph=gs.morphs.Box(size=CLAMP_SIZE, pos=pos, fixed=True),
        material=gs.materials.Rigid(needs_coup=False),
        surface=gs.surfaces.Plastic(color=color, roughness=0.55),
    )


def _clamp_center_left_of(left_edge_x: float) -> float:
    return left_edge_x - 0.5 * CLAMP_SIZE[0] - CLAMP_CLEARANCE


def _make_scene(*, gravity: tuple[float, float, float], res: tuple[int, int]) -> tuple[gs.Scene, Any]:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=gravity),
        fem_options=gs.options.FEMOptions(enable_vertex_constraints=True),
        coupler_options=gs.options.IPCCouplerOptions(contact_enable=False, newton_max_iterations=64),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=False,
    )
    _add_floor(scene)
    cam = scene.add_camera(
        res=res,
        pos=(0.0, -1.85, 0.82),
        lookat=(0.0, 0.0, 0.32),
        fov=44,
        GUI=False,
    )
    return scene, cam


def _record_scene(
    *,
    name: str,
    out_dir: Path,
    mesh_path: Path,
    frames: int,
    steps_per_frame: int,
    fps: int,
    res: tuple[int, int],
    setup_fn: Callable[[gs.Scene, Path], Callable[[], tuple[list[ConstraintProbe], Callable[[int], None]]]],
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.8),
) -> dict[str, Any]:
    scene, cam = _make_scene(gravity=gravity, res=res)
    post_build_fn = setup_fn(scene, mesh_path)
    scene.build()
    probes, update_fn = post_build_fn()

    frame_stds: list[float] = []
    max_errors: dict[str, float] = {probe.name: 0.0 for probe in probes}
    video_path = out_dir / f"{name}.mp4"

    cam.start_recording()
    for frame in range(frames):
        update_fn(frame)
        for _ in range(steps_per_frame):
            scene.step()

        rgb_array, *_ = cam.render(
            rgb=True,
            depth=False,
            segmentation=False,
            normal=False,
            colorize_seg=False,
            force_render=True,
        )
        frame_stds.append(_frame_std_normalized(rgb_array))

        for probe in probes:
            current = _entity_positions(probe.entity)[probe.verts_idx]
            target = probe.target_fn()
            error = np.linalg.norm(current - target, axis=1).max(initial=0.0)
            max_errors[probe.name] = max(max_errors[probe.name], float(error))

    cam.stop_recording(save_to_filename=video_path, fps=fps)

    video_size = video_path.stat().st_size if video_path.exists() else 0
    min_frame_std = min(frame_stds) if frame_stds else 0.0
    case_passed = (
        bool(video_size)
        and min_frame_std >= MIN_NORMALIZED_FRAME_STD
        and all(error <= MAX_ALLOWED_ERROR for error in max_errors.values())
    )
    return {
        "name": name,
        "video_path": str(video_path),
        "video_size_bytes": video_size,
        "frames": frames,
        "fps": fps,
        "steps_per_frame": steps_per_frame,
        "min_frame_std_normalized": min_frame_std,
        "mean_frame_std_normalized": float(np.mean(frame_stds)) if frame_stds else 0.0,
        "max_constraint_errors_m": max_errors,
        "passed": case_passed,
    }


def _setup_world_fixed(
    scene: gs.Scene, mesh_path: Path
) -> Callable[[], tuple[list[ConstraintProbe], Callable[[int], None]]]:
    fem = _add_fem_bar(scene, pos=FEM_POS)
    cloth = _add_cloth_sheet(scene, mesh_path=mesh_path, pos=CLOTH_POS_HIGH)

    def post_build() -> tuple[list[ConstraintProbe], Callable[[int], None]]:
        fem_idx = _side_indices(fem, axis=0, side="min")
        cloth_idx = _side_indices(cloth, axis=0, side="min", fraction=0.08)
        fem_target = _entity_positions(fem)[fem_idx].copy()
        cloth_target = _entity_positions(cloth)[cloth_idx].copy()
        fem.set_vertex_constraints(fem_idx, fem_target, stiffness=STIFFNESS)
        cloth.set_vertex_constraints(cloth_idx, cloth_target, stiffness=STIFFNESS)

        return [
            ConstraintProbe("fem_world_fixed_side", fem, fem_idx, lambda: fem_target),
            ConstraintProbe("cloth_world_fixed_edge", cloth, cloth_idx, lambda: cloth_target),
        ], lambda _frame: None

    return post_build


def _setup_moving_endpoint(
    scene: gs.Scene, mesh_path: Path
) -> Callable[[], tuple[list[ConstraintProbe], Callable[[int], None]]]:
    fem = _add_fem_bar(scene, pos=FEM_POS)
    cloth = _add_cloth_sheet(scene, mesh_path=mesh_path, pos=CLOTH_POS_LOW)

    def post_build() -> tuple[list[ConstraintProbe], Callable[[int], None]]:
        fem_left = _side_indices(fem, axis=0, side="min")
        fem_right = _side_indices(fem, axis=0, side="max")
        cloth_left = _side_indices(cloth, axis=0, side="min", fraction=0.08)
        cloth_right = _side_indices(cloth, axis=0, side="max", fraction=0.08)

        fem_left_target = _entity_positions(fem)[fem_left].copy()
        fem_right_target_0 = _entity_positions(fem)[fem_right].copy()
        cloth_left_target = _entity_positions(cloth)[cloth_left].copy()
        cloth_right_target_0 = _entity_positions(cloth)[cloth_right].copy()

        fem.set_vertex_constraints(fem_left, fem_left_target, stiffness=STIFFNESS)
        fem.set_vertex_constraints(fem_right, fem_right_target_0, stiffness=STIFFNESS)
        cloth.set_vertex_constraints(cloth_left, cloth_left_target, stiffness=STIFFNESS)
        cloth.set_vertex_constraints(cloth_right, cloth_right_target_0, stiffness=STIFFNESS)

        fem_right_target = fem_right_target_0.copy()
        cloth_right_target = cloth_right_target_0.copy()

        def update(frame: int) -> None:
            phase = frame / 39.0
            fem_offset = np.array([0.12 * phase, 0.02 * math.sin(math.pi * phase), 0.02], dtype=np.float64)
            cloth_offset = np.array([0.10 * phase, 0.0, 0.08 * math.sin(math.pi * phase)], dtype=np.float64)
            fem_right_target[:] = fem_right_target_0 + fem_offset
            cloth_right_target[:] = cloth_right_target_0 + cloth_offset
            fem.update_constraint_targets(fem_right, fem_right_target)
            cloth.update_constraint_targets(cloth_right, cloth_right_target)

        return [
            ConstraintProbe("fem_world_fixed_side", fem, fem_left, lambda: fem_left_target),
            ConstraintProbe("fem_moving_world_endpoint", fem, fem_right, lambda: fem_right_target),
            ConstraintProbe("cloth_world_fixed_edge", cloth, cloth_left, lambda: cloth_left_target),
            ConstraintProbe("cloth_moving_world_endpoint", cloth, cloth_right, lambda: cloth_right_target),
        ], update

    return post_build


def _setup_static_rigid_link(
    scene: gs.Scene, mesh_path: Path
) -> Callable[[], tuple[list[ConstraintProbe], Callable[[int], None]]]:
    fem = _add_fem_bar(scene, pos=FEM_POS)
    cloth = _add_cloth_sheet(scene, mesh_path=mesh_path, pos=CLOTH_POS_HIGH)
    fem_left_edge_x = FEM_POS[0] - 0.5 * FEM_BAR_SIZE[0]
    cloth_left_edge_x = CLOTH_POS_HIGH[0] - 0.5 * CLOTH_WIDTH
    fem_clamp = _add_clamp(
        scene,
        pos=(_clamp_center_left_of(fem_left_edge_x), FEM_POS[1], FEM_POS[2]),
        color=(0.15, 0.14, 0.13, 1.0),
    )
    cloth_clamp = _add_clamp(
        scene,
        pos=(_clamp_center_left_of(cloth_left_edge_x), CLOTH_POS_HIGH[1], CLOTH_POS_HIGH[2]),
        color=(0.15, 0.14, 0.13, 1.0),
    )

    def post_build() -> tuple[list[ConstraintProbe], Callable[[int], None]]:
        fem_idx = _side_indices(fem, axis=0, side="min")
        cloth_idx = _side_indices(cloth, axis=0, side="min", fraction=0.08)
        fem_target = _entity_positions(fem)[fem_idx].copy()
        cloth_target = _entity_positions(cloth)[cloth_idx].copy()
        fem.set_vertex_constraints(fem_idx, link=fem_clamp.base_link, stiffness=STIFFNESS)
        cloth.set_vertex_constraints(cloth_idx, link=cloth_clamp.base_link, stiffness=STIFFNESS)

        return [
            ConstraintProbe("fem_static_rigid_link", fem, fem_idx, lambda: fem_target),
            ConstraintProbe("cloth_static_rigid_link", cloth, cloth_idx, lambda: cloth_target),
        ], lambda _frame: None

    return post_build


def _setup_moving_rigid_link(
    scene: gs.Scene, mesh_path: Path
) -> Callable[[], tuple[list[ConstraintProbe], Callable[[int], None]]]:
    fem = _add_fem_bar(scene, pos=FEM_POS)
    cloth = _add_cloth_sheet(scene, mesh_path=mesh_path, pos=CLOTH_POS_HIGH)
    fem_left_edge_x = FEM_POS[0] - 0.5 * FEM_BAR_SIZE[0]
    cloth_left_edge_x = CLOTH_POS_HIGH[0] - 0.5 * CLOTH_WIDTH
    fem_clamp_pos_0 = np.array([_clamp_center_left_of(fem_left_edge_x), FEM_POS[1], FEM_POS[2]], dtype=np.float64)
    cloth_clamp_pos_0 = np.array(
        [_clamp_center_left_of(cloth_left_edge_x), CLOTH_POS_HIGH[1], CLOTH_POS_HIGH[2]], dtype=np.float64
    )
    fem_clamp = _add_clamp(scene, pos=tuple(fem_clamp_pos_0), color=(0.08, 0.55, 0.22, 1.0))
    cloth_clamp = _add_clamp(scene, pos=tuple(cloth_clamp_pos_0), color=(0.08, 0.55, 0.22, 1.0))

    def post_build() -> tuple[list[ConstraintProbe], Callable[[int], None]]:
        fem_idx = _side_indices(fem, axis=0, side="min")
        cloth_idx = _side_indices(cloth, axis=0, side="min", fraction=0.08)
        fem_target_0 = _entity_positions(fem)[fem_idx].copy()
        cloth_target_0 = _entity_positions(cloth)[cloth_idx].copy()
        fem_target = fem_target_0.copy()
        cloth_target = cloth_target_0.copy()
        fem.set_vertex_constraints(fem_idx, link=fem_clamp.base_link, stiffness=STIFFNESS)
        cloth.set_vertex_constraints(cloth_idx, link=cloth_clamp.base_link, stiffness=STIFFNESS)

        def update(frame: int) -> None:
            phase = frame / 39.0
            fem_disp = np.array([0.11 * phase, 0.03 * math.sin(math.pi * phase), 0.03], dtype=np.float64)
            cloth_disp = np.array([0.10 * phase, 0.02 * math.sin(math.pi * phase), 0.04], dtype=np.float64)
            fem_clamp.set_pos(tuple(fem_clamp_pos_0 + fem_disp), zero_velocity=True)
            cloth_clamp.set_pos(tuple(cloth_clamp_pos_0 + cloth_disp), zero_velocity=True)
            fem_target[:] = fem_target_0 + fem_disp
            cloth_target[:] = cloth_target_0 + cloth_disp

        return [
            ConstraintProbe("fem_moving_rigid_link", fem, fem_idx, lambda: fem_target),
            ConstraintProbe("cloth_moving_rigid_link", cloth, cloth_idx, lambda: cloth_target),
        ], update

    return post_build


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("code_agent/workspaces/suites/ipc_vertex_constraint_render_validation"),
    )
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--steps-per-frame", type=int, default=2)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--res", type=int, nargs=2, default=(512, 384), metavar=("W", "H"))
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = args.out_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    mesh_path = _write_cloth_grid_obj(assets_dir / "cloth_grid.obj")

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, precision="32", logging_level="warning")

    cases = [
        ("world_fixed_fem_cloth", _setup_world_fixed, (0.0, 0.0, -9.8)),
        ("moving_endpoint_fem_cloth", _setup_moving_endpoint, (0.0, 0.0, -2.0)),
        ("static_rigid_link_fem_cloth", _setup_static_rigid_link, (0.0, 0.0, -9.8)),
        ("moving_rigid_link_fem_cloth", _setup_moving_rigid_link, (0.0, 0.0, -3.0)),
    ]

    results = []
    try:
        for name, setup_fn, gravity in cases:
            results.append(
                _record_scene(
                    name=name,
                    out_dir=args.out_dir,
                    mesh_path=mesh_path,
                    frames=args.frames,
                    steps_per_frame=args.steps_per_frame,
                    fps=args.fps,
                    res=tuple(args.res),
                    setup_fn=setup_fn,
                    gravity=gravity,
                )
            )
    finally:
        gs.destroy()

    summary = {
        "backend": args.backend,
        "out_dir": str(args.out_dir),
        "thresholds": {
            "max_allowed_error_m": MAX_ALLOWED_ERROR,
            "min_normalized_frame_std": MIN_NORMALIZED_FRAME_STD,
        },
        "cases": results,
        "passed": all(case["passed"] for case in results),
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
