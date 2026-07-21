from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import trimesh

from code_agent.assets.mesh import remesh as remesh_module
from code_agent.assets.mesh.models import MeshManifoldCheckResult
from code_agent.assets.mesh.remesh import IsotropicRemeshConfig, remesh_mesh_asset
from code_agent.assets.mesh.texture.obj_io import rewrite_obj_mtllib, write_base_color_mtl
from code_agent.assets.mesh.texture.parameterization import parameterize_target_mesh_xatlas


def test_isotropic_remesh_reaches_target_and_transfers_texture(tmp_path: Path) -> None:
    source_mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    runtime_path = tmp_path / "source_runtime.obj"
    textured_path = tmp_path / "source_textured.obj"
    texture_path = tmp_path / "source_base_color.png"
    mtl_path = tmp_path / "source_textured.mtl"
    source_mesh.export(runtime_path)

    parameterize_target_mesh_xatlas(
        target_mesh=source_mesh,
        output_mesh_path=textured_path,
        texture_size=(64, 64),
    )
    write_base_color_mtl(mtl_path, texture_name=texture_path.name)
    rewrite_obj_mtllib(textured_path, mtl_name=mtl_path.name, material_name="material_0")
    _write_test_texture(texture_path)

    report = remesh_mesh_asset(
        IsotropicRemeshConfig(
            input_mesh_path=runtime_path,
            output_dir=tmp_path / "remeshed",
            target_face_count=400,
            target_face_tolerance=0.20,
            source_textured_mesh_path=textured_path,
            source_base_color_path=texture_path,
            validate_genesis=False,
        )
    )

    assert report["ok"] is True
    assert report["standalone"] is True
    assert report["pipeline_integrated"] is False
    assert report["source"]["face_count"] == 1280
    assert report["output"]["face_count"] < report["source"]["face_count"]
    assert report["target_check"]["relative_error"] <= 0.20
    assert report["manifold_validation"]["ok"] is True
    assert report["texture_transfer"]["ok"] is True
    assert report["texture_validation"]["ok"] is True
    assert report["texture_validation"]["visual_vertex_count"] > report["output"]["vertex_count"]
    assert report["texture_validation"]["seam_mapping"]["many_to_one"] is True
    assert report["texture_validation"]["seam_mapping"]["ok"] is True
    assert report["texture_validation"]["uv_face_count"] == report["output"]["face_count"]
    assert Path(report["artifacts"]["runtime_mesh"]).is_file()
    assert Path(report["artifacts"]["visual_mesh"]).is_file()
    assert Path(report["artifacts"]["base_color_texture"]).is_file()
    assert Path(report["artifacts"]["report"]).is_file()


def test_isotropic_remesh_requires_one_target_and_complete_texture_pair(tmp_path: Path) -> None:
    input_path = tmp_path / "mesh.obj"

    with pytest.raises(ValueError, match="exactly one"):
        IsotropicRemeshConfig(input_mesh_path=input_path, output_dir=tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        IsotropicRemeshConfig(
            input_mesh_path=input_path,
            output_dir=tmp_path,
            target_face_count=100,
            target_edge_length=0.1,
        )
    with pytest.raises(ValueError, match="must be supplied together"):
        IsotropicRemeshConfig(
            input_mesh_path=input_path,
            output_dir=tmp_path,
            target_face_count=100,
            source_textured_mesh_path=tmp_path / "visual.obj",
        )


def test_isotropic_remesh_stops_before_texture_and_genesis_after_preflight_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    runtime_path = tmp_path / "source.obj"
    source_mesh.export(runtime_path)

    def fail_manifold(mesh_path: Path) -> MeshManifoldCheckResult:
        return MeshManifoldCheckResult(
            ok=False,
            mesh_path=mesh_path,
            vertex_count=100,
            face_count=200,
            component_count=1,
            is_watertight=False,
            is_winding_consistent=True,
            volume=None,
            tetgen_ready=False,
            error="synthetic topology failure",
        )

    def unexpected_validation(*args: object, **kwargs: object) -> None:
        raise AssertionError("Genesis validation must be skipped after preflight failure.")

    monkeypatch.setattr(remesh_module, "run_mesh_manifold_check", fail_manifold)
    monkeypatch.setattr(remesh_module, "run_genesis_rigid_import_validation", unexpected_validation)
    monkeypatch.setattr(remesh_module, "run_genesis_fem_import_validation", unexpected_validation)
    monkeypatch.setattr(remesh_module, "run_genesis_cloth_import_validation", unexpected_validation)

    report = remesh_mesh_asset(
        IsotropicRemeshConfig(
            input_mesh_path=runtime_path,
            output_dir=tmp_path / "failed",
            target_face_count=200,
            validate_genesis=True,
        )
    )

    assert report["ok"] is False
    assert report["failure_stage"] == "manifold_validation"
    assert report["texture_validation"]["skipped"] is True
    assert report["genesis_fem_import_validation"]["skipped"] is True
    assert report["artifacts"]["visual_mesh"] is None


def _write_test_texture(path: Path) -> None:
    size = 64
    x = np.linspace(0, 255, size, dtype=np.uint8)
    y = np.linspace(255, 0, size, dtype=np.uint8)
    red = np.broadcast_to(x[None, :], (size, size))
    green = np.broadcast_to(y[:, None], (size, size))
    blue = np.where((np.indices((size, size)).sum(axis=0) // 8) % 2 == 0, 32, 224).astype(np.uint8)
    alpha = np.full((size, size), 255, dtype=np.uint8)
    Image.fromarray(np.stack((red, green, blue, alpha), axis=-1), mode="RGBA").save(path)
