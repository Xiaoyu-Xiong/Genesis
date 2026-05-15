from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from code_agent.configs import CONFIGS


def render_xml_preview(
    xml_path: Path,
    output_dir: Path,
    *,
    res: tuple[int, int] = CONFIGS.xml_asset.preview_res,
) -> dict[str, Any]:
    """Render static MuJoCo preview views for visual inspection of a generated XML asset."""

    return _render_xml_preview_in_subprocess(xml_path.resolve(), output_dir.resolve(), res=res)


def _render_xml_preview_in_subprocess(xml_path: Path, output_dir: Path, *, res: tuple[int, int]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "preview_report.json"
    report_path.unlink(missing_ok=True)
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    command = [
        sys.executable,
        "-m",
        "code_agent.assets.xml.preview",
        "--worker",
        str(xml_path),
        str(output_dir),
        str(int(res[0])),
        str(int(res[1])),
        str(report_path),
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report = _base_report(xml_path, output_dir)
            report["errors"].append(f"Preview worker wrote invalid JSON: {exc}")
    else:
        report = _base_report(xml_path, output_dir)

    report["worker"] = {
        "command": command,
        "returncode": completed.returncode,
        "env_overrides": {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
        },
    }
    if completed.stdout:
        report["worker"]["stdout"] = completed.stdout
    if completed.stderr:
        report["worker"]["stderr"] = completed.stderr
    if completed.returncode != 0:
        report["ok"] = False
        report["errors"].append(f"Preview worker exited with status {completed.returncode}.")
    return report


def _render_xml_preview_impl(
    xml_path: Path,
    output_dir: Path,
    *,
    res: tuple[int, int],
) -> dict[str, Any]:
    xml_path = xml_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = _base_report(xml_path, output_dir)
    report["headless_env"] = {
        "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
        "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
    }

    try:
        import mujoco
        import numpy as np
        from PIL import Image
    except Exception as exc:
        report["errors"].append(f"Preview imports failed: {type(exc).__name__}: {exc}")
        return report

    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        requested_width, requested_height = int(res[0]), int(res[1])
        offwidth = int(getattr(model.vis.global_, "offwidth", requested_width))
        offheight = int(getattr(model.vis.global_, "offheight", requested_height))
        width = min(requested_width, offwidth) if offwidth > 0 else requested_width
        height = min(requested_height, offheight) if offheight > 0 else requested_height
        if (width, height) != (requested_width, requested_height):
            report["warnings"].append(
                f"Preview resolution reduced from {requested_width}x{requested_height} to {width}x{height} "
                "to fit the MJCF offscreen framebuffer."
            )
        renderer = mujoco.Renderer(model, height=height, width=width)
    except Exception as exc:
        report["errors"].append(f"MuJoCo preview setup failed: {type(exc).__name__}: {exc}")
        return report

    center = [float(value) for value in model.stat.center]
    extent = max(float(model.stat.extent), 0.25)
    distance = max(0.5, CONFIGS.xml_asset.preview_distance_scale * extent)
    report["model_summary"] = {
        "stat_center": center,
        "stat_extent": extent,
        "camera_distance": distance,
        "resolution": [width, height],
    }

    views = {
        "front": (0.0, -18.0),
        "side": (90.0, -18.0),
        "iso": (45.0, -25.0),
        "top": (0.0, -89.0),
    }
    try:
        mujoco.mj_forward(model, data)
        for name, (azimuth, elevation) in views.items():
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.lookat[:] = center
            camera.distance = distance
            camera.azimuth = azimuth
            camera.elevation = elevation
            renderer.update_scene(data, camera)
            image = renderer.render()
            image_path = output_dir / f"{name}.png"
            Image.fromarray(image).save(image_path)
            stats = _image_stats(np.asarray(image), image_path)
            report["views"].append(
                {
                    "name": name,
                    "image_path": str(image_path.resolve()),
                    "azimuth": azimuth,
                    "elevation": elevation,
                    **stats,
                }
            )
    except Exception as exc:
        report["errors"].append(f"MuJoCo preview render failed: {type(exc).__name__}: {exc}")
    finally:
        close = getattr(renderer, "close", None)
        if callable(close):
            close()

    nonblank_views = [view for view in report["views"] if view.get("nonblank")]
    if len(nonblank_views) < 2:
        report["errors"].append("Preview produced fewer than two nonblank views.")
    report["ok"] = not report["errors"]
    return report


def _base_report(xml_path: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "ok": False,
        "xml_path": str(xml_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "renderer": "mujoco.Renderer",
        "views": [],
        "errors": [],
        "warnings": [],
    }


def _image_stats(image, image_path: Path) -> dict[str, Any]:
    mean = float(image.mean())
    std = float(image.std())
    height = int(image.shape[0])
    width = int(image.shape[1])
    # A valid asset can occupy a small part of the image, so keep the threshold intentionally permissive.
    nonblank = std > 1e-3 and image_path.exists() and image_path.stat().st_size > 0
    return {
        "width": width,
        "height": height,
        "mean": mean,
        "std": std,
        "nonblank": nonblank,
    }


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("xml_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("width", type=int)
    parser.add_argument("height", type=int)
    parser.add_argument("report_path", type=Path)
    args = parser.parse_args()
    if not args.worker:
        raise SystemExit("preview.py is an internal XML preview worker; pass --worker.")

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    report = _render_xml_preview_impl(args.xml_path, args.output_dir, res=(args.width, args.height))
    args.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    _main()
