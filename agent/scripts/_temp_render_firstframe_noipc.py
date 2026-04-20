from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.utils.tools import save_img_arr


def _reset_genesis() -> None:
    try:
        gs.destroy()
    except Exception:
        pass


def _add_rigid_entity(scene, body: dict) -> None:
    shape = body["shape"]
    pose = body["initial_pose"]
    kind = shape["kind"]
    material = gs.materials.Rigid()

    if kind == "box":
        scene.add_entity(
            morph=gs.morphs.Box(size=tuple(shape["size"]), pos=tuple(pose["pos"]), quat=tuple(pose["quat"])),
            material=material,
            name=body["name"],
        )
    elif kind == "cylinder":
        scene.add_entity(
            morph=gs.morphs.Cylinder(
                radius=float(shape["radius"]),
                height=float(shape["height"]),
                pos=tuple(pose["pos"]),
                quat=tuple(pose["quat"]),
            ),
            material=material,
            name=body["name"],
        )
    elif kind == "plane":
        scene.add_entity(
            morph=gs.morphs.Plane(pos=tuple(pose["pos"]), quat=tuple(pose["quat"])),
            material=material,
            name=body["name"],
        )
    elif kind == "mjcf":
        scene.add_entity(
            morph=gs.morphs.MJCF(
                file=shape["file"],
                scale=float(shape.get("scale", 1.0)),
                pos=tuple(pose["pos"]),
                quat=tuple(pose["quat"]),
            ),
            material=material,
            name=body["name"],
        )


def _add_deformable_entity(scene, body: dict) -> None:
    shape = body["shape"]
    pose = body["initial_pose"]
    mat = body["deformable_material"]
    material = gs.materials.FEM.Elastic(
        rho=float(mat["rho"]),
        E=float(mat["E"]),
        nu=float(mat["nu"]),
    )

    if shape["kind"] == "mesh":
        morph = gs.morphs.Mesh(
            file=shape["file"],
            scale=float(shape["scale"]),
            pos=tuple(pose["pos"]),
            quat=tuple(pose["quat"]),
        )
    elif shape["kind"] == "box":
        morph = gs.morphs.Box(
            size=tuple(shape["size"]),
            pos=tuple(pose["pos"]),
            quat=tuple(pose["quat"]),
        )
    elif shape["kind"] == "sphere":
        morph = gs.morphs.Sphere(
            radius=float(shape["radius"]),
            pos=tuple(pose["pos"]),
            quat=tuple(pose["quat"]),
        )
    elif shape["kind"] == "cylinder":
        morph = gs.morphs.Cylinder(
            radius=float(shape["radius"]),
            height=float(shape["height"]),
            pos=tuple(pose["pos"]),
            quat=tuple(pose["quat"]),
        )
    else:
        raise ValueError(f"Unsupported deformable shape kind for no-IPC first-frame render: {shape['kind']}")

    scene.add_entity(
        morph=morph,
        material=material,
        name=body["name"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.ir.read_text(encoding="utf-8"))
    render = payload["scene"]["render"]

    _reset_genesis()
    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(
            dt=float(payload["scene"]["sim"]["dt"]),
            gravity=tuple(payload["scene"]["sim"]["gravity"]),
        ),
        fem_options=gs.options.FEMOptions(),
        renderer=gs.renderers.Rasterizer(),
        viewer_options=gs.options.ViewerOptions(
            res=tuple(render["res"]),
            camera_pos=tuple(render["camera_pos"]),
            camera_lookat=tuple(render["camera_lookat"]),
            camera_fov=float(render["camera_fov"]),
        ),
        vis_options=gs.options.VisOptions(
            ambient_light=(0.85, 0.85, 0.85),
            show_world_frame=False,
            show_link_frame=False,
            plane_reflection=False,
        ),
    )

    if payload["scene"].get("add_ground", False):
        scene.add_entity(morph=gs.morphs.Plane(), material=gs.materials.Rigid(), name="ground")

    for body in payload["bodies"]:
        if body["simulation_kind"] == "rigid":
            _add_rigid_entity(scene, body)
        elif body["simulation_kind"] == "deformable":
            _add_deformable_entity(scene, body)

    camera = scene.add_camera(
        res=tuple(render["res"]),
        pos=tuple(render["camera_pos"]),
        lookat=tuple(render["camera_lookat"]),
        up=tuple(render["camera_up"]),
        fov=float(render["camera_fov"]),
        GUI=False,
    )

    scene.build()
    rgb, _, _, _ = camera.render(rgb=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_img_arr(np.asarray(rgb), str(args.out))
    print(args.out)
    scene.destroy()
    _reset_genesis()


if __name__ == "__main__":
    main()
