from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from shutil import copyfile

import igl
import numpy as np

import genesis as gs
import genesis.utils.element as eu
import genesis.utils.mesh as mu
from genesis.utils.tools import save_img_arr

from agent.configs import CONFIGS
from agent.io_utils import dump_json
from agent.mesh.texture_transfer import transfer_texture_to_repaired_mesh


DEFAULT_CASES = (
    "banana",
    "baseball_cap",
    "gift_box",
    "rubber_duck",
)


def write_textured_obj(
    *,
    obj_path: Path,
    verts: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    texture_path: Path,
) -> None:
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = obj_path.with_suffix(".mtl")
    copied_tex_path = obj_path.parent / "base_color.png"
    copyfile(texture_path, copied_tex_path)

    lines = [f"mtllib {mtl_path.name}"]
    for vertex in verts:
        lines.append(f"v {float(vertex[0]):.9f} {float(vertex[1]):.9f} {float(vertex[2]):.9f}")
    lines.append("usemtl material_0")
    for uv in uvs:
        lines.append(f"vt {float(uv[0]):.9f} {float(uv[1]):.9f}")
    for face in faces:
        a, b, c = int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1
        lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    mtl_path.write_text(
        "\n".join(
            [
                "newmtl material_0",
                "Ka 0.000000 0.000000 0.000000",
                "Kd 1.000000 1.000000 1.000000",
                "Ks 0.000000 0.000000 0.000000",
                "d 1.0",
                "illum 2",
                f"map_Kd {copied_tex_path.name}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _normalized(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return np.array((1.0, 0.0, 0.0), dtype=np.float64)
    return arr / norm


def _camera_distance(*, diagonal: float, fov: float) -> float:
    radius = max(diagonal * 0.5, 0.1)
    fov_rad = math.radians(max(1.0, min(170.0, fov)))
    return max((radius / math.tan(fov_rad * 0.5)) * 1.5, diagonal * 1.2, 0.75)


def render_views(
    *,
    mesh_path: Path,
    out_dir: Path,
    res: tuple[int, int] = (768, 768),
    fov: float = 35.0,
) -> dict[str, str]:
    import trimesh

    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load_mesh(str(mesh_path), force="mesh", skip_texture=True, process=False)
    bounds = mesh.bounds.astype(np.float64)
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
            ambient_light=(0.85, 0.85, 0.85),
            show_world_frame=False,
            show_link_frame=False,
            plane_reflection=False,
        ),
    )

    scene.add_entity(
        morph=gs.morphs.Plane(pos=(0.0, 0.0, -0.002)),
        surface=gs.surfaces.Rough(color=(0.92, 0.92, 0.92, 1.0)),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(mesh_path),
            scale=1.0,
            pos=translation,
            collision=False,
        )
    )
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
            "pos": tuple((focus + _normalized((1.0, -1.0, 0.75)) * distance).tolist()),
            "lookat": tuple(focus.tolist()),
            "up": (0.0, 0.0, 1.0),
        },
    }

    outputs: dict[str, str] = {}
    for name, spec in view_specs.items():
        camera.set_pose(pos=spec["pos"], lookat=spec["lookat"], up=spec["up"])
        rgb_arr, _, _, _ = camera.render(rgb=True)
        out_path = out_dir / f"{name}.png"
        save_img_arr(np.asarray(rgb_arr), str(out_path))
        outputs[name] = str(out_path)
    return outputs


def run_case(case_dir: Path, tet_resolution: int, render_res: tuple[int, int], render_fov: float) -> dict[str, object]:
    source_mesh_path = case_dir / "processed" / "repaired.obj"
    source_tex_path = case_dir / "processed" / "base_color.png"
    raw_textured_mesh_path = case_dir / "textured" / "model.obj"
    raw_textured_tex_path = case_dir / "textured" / "base_color.png"
    debug_root = case_dir / "main_pipeline_texture_debug"
    remesh_stage_dir = debug_root / "remesh_stage"
    tet_stage_dir = debug_root / "tet_boundary_stage"

    source_mesh = mu.load_mesh(source_mesh_path)
    feature_sizes = tuple(np.maximum(source_mesh.bounding_box.extents.astype(np.float64), 1e-6).tolist())
    target_edge = max(float(min(feature_sizes)), 1e-6) / float(2 * tet_resolution + 1)

    remeshed_geom = mu.remesh_surface_mesh(source_mesh, edge_len_abs=target_edge, fix=False)
    remesh_raw_obj = remesh_stage_dir / "processed" / "remeshed_raw.obj"
    remesh_raw_obj.parent.mkdir(parents=True, exist_ok=True)
    remeshed_geom.export(remesh_raw_obj)

    remesh_source_mesh = raw_textured_mesh_path if raw_textured_mesh_path.exists() else source_mesh_path
    remesh_source_tex = raw_textured_tex_path if raw_textured_tex_path.exists() else source_tex_path
    alignment_translation = None
    repair_json_path = case_dir / "repair.json"
    if remesh_source_mesh == raw_textured_mesh_path and repair_json_path.exists():
        repair_payload = json.loads(repair_json_path.read_text(encoding="utf-8"))
        centroid = repair_payload.get("centroid_before_translation")
        if centroid is not None:
            alignment_translation = tuple(float(value) for value in centroid)

    remesh_transfer = transfer_texture_to_repaired_mesh(
        source_mesh_path=remesh_source_mesh,
        source_base_color_path=remesh_source_tex,
        target_mesh_path=remesh_raw_obj,
        output_dir=remesh_stage_dir,
        alignment_translation=alignment_translation,
    )

    remesh_views = render_views(
        mesh_path=remesh_raw_obj,
        out_dir=remesh_stage_dir / "render_views",
        res=render_res,
        fov=render_fov,
    )

    verts, elems, uvs = eu.mesh_to_elements(
        file=source_mesh_path,
        pos=(0.0, 0.0, 0.0),
        scale=1.0,
        tet_cfg={"tet_resolution": tet_resolution},
    )
    if uvs is None:
        raise RuntimeError("mesh_to_elements returned uvs=None.")

    render_artifact = eu.get_mesh_to_elements_render_artifact(
        source_mesh_path,
        1.0,
        {"tet_resolution": tet_resolution},
    )
    if not isinstance(render_artifact, dict) or not render_artifact:
        raise RuntimeError("Main pipeline did not register a seam-aware render artifact.")
    render_src_indices = np.asarray(render_artifact["render_vertex_src_indices"], dtype=np.int64)
    render_faces = np.asarray(render_artifact["render_faces"], dtype=np.int64)
    render_uvs = np.asarray(render_artifact["render_uvs"], dtype=np.float64)
    tet_boundary_texture_path = render_artifact.get("texture_path")
    if tet_boundary_texture_path is None:
        raise RuntimeError("Main pipeline did not register a tet-boundary texture path.")
    render_verts = np.asarray(verts, dtype=np.float64)[render_src_indices]

    tet_boundary_obj = tet_stage_dir / "boundary_textured.obj"
    write_textured_obj(
        obj_path=tet_boundary_obj,
        verts=render_verts,
        faces=render_faces,
        uvs=render_uvs,
        texture_path=Path(tet_boundary_texture_path),
    )

    tet_boundary_views = render_views(
        mesh_path=tet_boundary_obj,
        out_dir=tet_stage_dir / "render_views",
        res=render_res,
        fov=render_fov,
    )

    result = {
        "source_mesh_path": str(source_mesh_path),
        "source_texture_path": str(source_tex_path),
        "raw_textured_mesh_path": str(raw_textured_mesh_path) if raw_textured_mesh_path.exists() else None,
        "raw_textured_texture_path": str(raw_textured_tex_path) if raw_textured_tex_path.exists() else None,
        "tet_resolution": tet_resolution,
        "target_edge": target_edge,
        "remesh_stage": {
            "raw_obj": str(remesh_raw_obj),
            "transfer": remesh_transfer.to_dict(),
            "views": remesh_views,
            "vertex_count": int(len(remeshed_geom.vertices)),
            "face_count": int(len(remeshed_geom.faces)),
        },
        "tet_boundary_stage": {
            "boundary_obj": str(tet_boundary_obj),
            "texture_path": str(tet_boundary_texture_path),
            "views": tet_boundary_views,
            "tet_vertex_count": int(len(verts)),
            "tet_element_count": int(len(elems)),
            "boundary_vertex_count": int(len(render_verts)),
            "boundary_face_count": int(len(render_faces)),
        },
    }
    dump_json(result, debug_root / "summary.json")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug main-pipeline texture stages for one or more existing textured mesh cases."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("agent/runs/mesh_meshy_texture_suite/20260414_005158"),
        help="Root run directory containing per-case subdirectories.",
    )
    parser.add_argument(
        "--cases",
        type=str,
        nargs="*",
        default=list(DEFAULT_CASES),
        help="Case directory names under --run-root.",
    )
    parser.add_argument(
        "--tet-resolution",
        type=int,
        default=CONFIGS.deformable.tet_resolution,
        help="Tet resolution used to mimic the main deformable pipeline.",
    )
    parser.add_argument(
        "--render-res",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=(768, 768),
        help="Output render resolution.",
    )
    parser.add_argument("--render-fov", type=float, default=35.0, help="Camera field of view in degrees.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    gs.init(backend=gs.cpu, logging_level="warning")

    render_res = tuple(args.render_res)
    summary: dict[str, object] = {"run_root": str(args.run_root), "cases": {}}

    for case_name in args.cases:
        case_dir = args.run_root / case_name
        summary["cases"][case_name] = run_case(
            case_dir=case_dir,
            tet_resolution=int(args.tet_resolution),
            render_res=render_res,
            render_fov=float(args.render_fov),
        )

    dump_json(summary, args.run_root / "main_pipeline_texture_debug_summary.json")


if __name__ == "__main__":
    main()
