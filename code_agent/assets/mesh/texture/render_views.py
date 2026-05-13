from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh

import genesis as gs
from genesis.utils.tools import save_img_arr

from code_agent.configs import CONFIGS


def render_textured_mesh_views(
    *,
    mesh_path: str | Path,
    out_dir: str | Path,
    texture_path: str | Path | None = None,
    visual_mesh_path: str | Path | None = None,
    scale: float | tuple[float, float, float] | list[float] = 1.0,
    file_meshes_are_zup: bool = False,
    backend: str = CONFIGS.harness.default_backend,
    res: tuple[int, int] = (768, 768),
    fov: float = 35.0,
) -> dict[str, str]:
    mesh_path = Path(mesh_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load_mesh(str(mesh_path), force="mesh", skip_texture=True, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected textured mesh path to load as Trimesh, got {type(mesh).__name__}")

    scale_vec = _scale_vector(scale)
    bounds = _bounds_in_genesis_frame(mesh.bounds.astype(np.float64, copy=False), file_meshes_are_zup)
    bounds = bounds * scale_vec
    bbox_min = bounds[0]
    bbox_max = bounds[1]
    extents = np.maximum(bbox_max - bbox_min, 1e-6)
    center_xy = 0.5 * (bbox_min[:2] + bbox_max[:2])
    translation = (-float(center_xy[0]), -float(center_xy[1]), -float(bbox_min[2]))
    height = float(extents[2])
    diagonal = float(np.linalg.norm(extents))
    focus = np.array((0.0, 0.0, max(0.4 * height, 0.05)), dtype=np.float64)
    distance = _camera_distance(diagonal=diagonal, fov=fov)
    z_boost = max(0.15 * height, 0.05)

    gs_backend = gs.cpu if backend == "cpu" else gs.gpu
    gs.destroy()
    gs.init(backend=gs_backend, logging_level="info")
    scene = None
    try:
        scene = gs.Scene(
            show_viewer=False,
            renderer=gs.renderers.Rasterizer(),
            viewer_options=gs.options.ViewerOptions(
                res=res,
                camera_pos=(distance, -distance, distance),
                camera_lookat=tuple(focus.tolist()),
                camera_fov=fov,
            ),
            vis_options=gs.options.VisOptions(
                ambient_light=(0.9, 0.9, 0.9),
                background_color=(1.0, 1.0, 1.0),
                show_world_frame=False,
                show_link_frame=False,
                plane_reflection=False,
            ),
        )

        scene.add_entity(
            morph=gs.morphs.Plane(pos=(0.0, 0.0, -0.002)),
            surface=gs.surfaces.Rough(color=(0.96, 0.96, 0.96, 1.0)),
        )
        mesh_kwargs = {
            "file": str(mesh_path),
            "scale": scale_vec.tolist(),
            "pos": translation,
            "file_meshes_are_zup": file_meshes_are_zup,
            "collision": False,
            "convexify": False,
        }
        visual_mesh_path = Path(visual_mesh_path) if visual_mesh_path is not None else None
        if visual_mesh_path is not None and visual_mesh_path.is_file():
            mesh_kwargs["visual_file"] = str(visual_mesh_path)
            mesh_kwargs["visual_file_meshes_are_zup"] = file_meshes_are_zup
        mesh_morph = gs.morphs.Mesh(**mesh_kwargs)

        texture_path = Path(texture_path) if texture_path is not None else None
        if visual_mesh_path is not None:
            scene.add_entity(morph=mesh_morph)
        elif texture_path is not None and texture_path.is_file():
            scene.add_entity(
                morph=mesh_morph,
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ImageTexture(image_path=str(texture_path)),
                ),
            )
        else:
            scene.add_entity(morph=mesh_morph, surface=gs.surfaces.Rough(color=(0.72, 0.64, 0.92, 1.0)))
        camera = scene.add_camera(
            res=res,
            pos=(distance, -distance, distance),
            lookat=tuple(focus.tolist()),
            fov=fov,
            GUI=False,
        )
        scene.build()

        view_specs = {
            "front": {
                "pos": (0.0, -distance, focus[2] + z_boost),
                "lookat": tuple(focus.tolist()),
                "up": (0.0, 0.0, 1.0),
            },
            "side": {
                "pos": (distance, 0.0, focus[2] + z_boost),
                "lookat": tuple(focus.tolist()),
                "up": (0.0, 0.0, 1.0),
            },
            "top": {
                "pos": (0.0, 0.0, distance),
                "lookat": tuple(focus.tolist()),
                "up": (0.0, 1.0, 0.0),
            },
            "iso": {
                "pos": tuple((focus + _normalized(np.array((1.0, -1.0, 0.75))) * distance).tolist()),
                "lookat": tuple(focus.tolist()),
                "up": (0.0, 0.0, 1.0),
            },
        }

        outputs: dict[str, str] = {}
        for view_name, spec in view_specs.items():
            camera.set_pose(pos=spec["pos"], lookat=spec["lookat"], up=spec["up"])
            rgb_arr, _, _, _ = camera.render(rgb=True, depth=False, segmentation=False, normal=False, force_render=True)
            out_path = out_dir / f"{view_name}.png"
            save_img_arr(np.asarray(rgb_arr), str(out_path))
            outputs[view_name] = str(out_path)

        return outputs
    finally:
        if scene is not None:
            scene.destroy()
        gs.destroy()


def _bounds_in_genesis_frame(bounds: np.ndarray, file_meshes_are_zup: bool) -> np.ndarray:
    if file_meshes_are_zup:
        return bounds
    bbox_min = bounds[0]
    bbox_max = bounds[1]
    return np.array(
        (
            (bbox_min[0], -bbox_max[2], bbox_min[1]),
            (bbox_max[0], -bbox_min[2], bbox_max[1]),
        ),
        dtype=np.float64,
    )


def _camera_distance(*, diagonal: float, fov: float) -> float:
    radius = max(diagonal * 0.5, 0.1)
    fov_rad = math.radians(max(1.0, min(170.0, fov)))
    return max((radius / math.tan(fov_rad * 0.5)) * 1.5, diagonal * 1.2, 0.75)


def _scale_vector(scale: float | tuple[float, float, float] | list[float]) -> np.ndarray:
    scale_arr = np.atleast_1d(np.asarray(scale, dtype=np.float64))
    if scale_arr.size == 1:
        return np.repeat(float(scale_arr[0]), 3)
    if scale_arr.size == 3:
        return scale_arr.reshape(3)
    raise ValueError(f"scale must be a scalar or length-3 vector, got shape {scale_arr.shape}")


def _normalized(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return np.array((1.0, 0.0, 0.0), dtype=np.float64)
    return vec / norm
