import math

import numpy as np
import pytest
import trimesh

import genesis.utils.element as eu


def _capture_primitive_meshing(monkeypatch):
    calls: dict[str, list[dict[str, object]]] = {"remesh": [], "tetrahedralize": []}

    def fake_remesh_surface_mesh(mesh, edge_len_abs=None, edge_len_ratio=0.01, fix=True):
        calls["remesh"].append(
            {
                "mesh": mesh.copy(),
                "edge_len_abs": edge_len_abs,
                "edge_len_ratio": edge_len_ratio,
                "fix": fix,
            }
        )
        return mesh

    def fake_tetrahedralize_mesh(mesh, tet_cfg):
        calls["tetrahedralize"].append({"mesh": mesh.copy(), "tet_cfg": dict(tet_cfg)})
        return np.zeros((4, 3), dtype=np.float64), np.zeros((1, 4), dtype=np.int32)

    monkeypatch.setattr(eu.mu, "remesh_surface_mesh", fake_remesh_surface_mesh)
    monkeypatch.setattr(eu.mu, "tetrahedralize_mesh", fake_tetrahedralize_mesh)
    return calls


def test_deformable_primitive_box_uses_boosted_target_edge(monkeypatch):
    calls = _capture_primitive_meshing(monkeypatch)

    eu.box_to_elements(size=(1.0, 1.0, 1.0), tet_cfg={"tet_resolution": 2})

    target_edge = 1.0 / 7.0
    assert calls["remesh"][0]["edge_len_abs"] == pytest.approx(target_edge)
    assert calls["remesh"][0]["fix"] is False
    tet_cfg = calls["tetrahedralize"][0]["tet_cfg"]
    assert tet_cfg["maxvolume"] == pytest.approx((target_edge**3) * math.sqrt(2.0) / 12.0)


def test_deformable_primitive_sphere_uses_boosted_subdivisions(monkeypatch):
    calls = _capture_primitive_meshing(monkeypatch)

    eu.sphere_to_elements(radius=0.5, tet_cfg={"tet_resolution": 2})

    mesh = calls["remesh"][0]["mesh"]
    assert calls["remesh"][0]["edge_len_abs"] == pytest.approx(1.0 / 7.0)
    assert len(mesh.faces) == 20 * 4**3


def test_deformable_primitive_cylinder_uses_higher_minimum_sections(monkeypatch):
    calls = _capture_primitive_meshing(monkeypatch)

    eu.cylinder_to_elements(radius=0.5, height=1.0, tet_cfg={"tet_resolution": 2})

    mesh = calls["remesh"][0]["mesh"]
    radii = np.linalg.norm(np.asarray(mesh.vertices)[:, :2], axis=1)
    angles = np.round(np.mod(np.arctan2(mesh.vertices[:, 1], mesh.vertices[:, 0]), 2.0 * math.pi), decimals=6)
    section_count = len(np.unique(angles[radii > 0.49]))
    assert calls["remesh"][0]["edge_len_abs"] == pytest.approx(1.0 / 7.0)
    assert section_count == 32


def test_mesh_file_tetrahedralization_keeps_unboosted_tet_resolution(tmp_path, monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_tetrahedralize_mesh(mesh, tet_cfg):
        calls.append({"mesh": mesh.copy(), "tet_cfg": dict(tet_cfg)})
        return np.zeros((4, 3), dtype=np.float64), np.zeros((1, 4), dtype=np.int32)

    mesh_path = tmp_path / "box.obj"
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(mesh_path)

    monkeypatch.setattr(eu.mu, "get_tet_path", lambda *_args, **_kwargs: str(tmp_path / "box.tet"))
    monkeypatch.setattr(eu.mu, "tetrahedralize_mesh", fake_tetrahedralize_mesh)

    eu.mesh_to_elements(mesh_path, tet_cfg={"tet_resolution": 2})

    target_edge = 1.0 / 5.0
    assert calls[0]["tet_cfg"]["maxvolume"] == pytest.approx((target_edge**3) * math.sqrt(2.0) / 12.0)
