from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

from code_agent.assets.mesh import episode as mesh_episode
from code_agent.assets.mesh.manifest import _scale_to_bbox, _uniform_scale_factor
from code_agent.assets.mesh.models import MeshGenesisFEMImportResult
from code_agent.assets.mesh.request_adapter import request_size
from code_agent.configs import deformable_config_dict
from code_agent.planner.session import PlannerSession, PlannerSessionConfig
from code_agent.utils import codex
from code_agent.utils.integrator import write_main


def test_run_codex_exec_resolves_relative_io_paths_against_repo_root(tmp_path, monkeypatch):
    fake_repo = tmp_path / "repo"
    polluted_cwd = tmp_path / "polluted"
    fake_repo.mkdir()
    polluted_cwd.mkdir()
    monkeypatch.setattr(codex, "DEFAULT_REPO_ROOT", fake_repo)
    monkeypatch.chdir(polluted_cwd)

    result = codex.run_codex_exec(
        codex.CodexExecRequest(
            role="test",
            prompt="",
            cwd=Path("."),
            output_jsonl_path=Path("logs/codex_test.jsonl"),
            final_message_path=Path("logs/codex_test.final.txt"),
            codex_bin="definitely_missing_codex_binary_for_test",
        )
    )

    expected_jsonl = fake_repo / "logs" / "codex_test.jsonl"
    expected_final = fake_repo / "logs" / "codex_test.final.txt"
    assert result.output_jsonl_path == str(expected_jsonl.resolve())
    assert result.final_message_path == str(expected_final.resolve())
    assert result.cwd == str(fake_repo.resolve())
    assert expected_jsonl.is_file()
    assert expected_final.is_file()
    assert not (polluted_cwd / "logs" / "codex_test.jsonl").exists()


def test_planner_schema_lookup_is_repo_relative_when_cwd_is_polluted(tmp_path, monkeypatch):
    polluted_cwd = tmp_path / "polluted"
    polluted_cwd.mkdir()
    monkeypatch.chdir(polluted_cwd)
    session = PlannerSession(
        PlannerSessionConfig(
            case_id="case",
            task="task",
            case_dir=tmp_path / "case",
            backend="gpu",
            timeout_sec=1.0,
            render=False,
            repair_rounds=0,
        )
    )

    schema = session.load_json(Path("code_agent/specs/planner_action.schema.json"))
    errors = session.validate_json_schema({}, Path("code_agent/specs/planner_action.schema.json"))

    assert schema is not None
    assert isinstance(errors, list)
    assert not any("schema missing" in error for error in errors)


def test_deformable_config_does_not_override_fem_friction_mu():
    cfg = deformable_config_dict(deformable_enabled=True, ipc_enabled=True)

    assert "fem_friction_mu" not in cfg
    assert "friction" in cfg


def test_generated_main_adds_repo_root_before_code_agent_import(tmp_path):
    run_dir = tmp_path / "case"
    main_path = write_main(
        run_dir=run_dir,
        task="task",
        default_steps=1,
        default_render_fps=1,
        default_duration_sec=1.0,
        default_target_video_frames=1,
        deformable_cfg={"ipc_contact_d_hat_adaptive": True},
    )

    source = main_path.read_text(encoding="utf-8")
    sys_path_pos = source.index("sys.path.insert(0, str(REPO_ROOT))")
    adaptive_import_pos = source.index("from code_agent.utils.adaptive_ipc import")

    assert sys_path_pos < adaptive_import_pos


def test_generated_mesh_manifest_uses_uniform_bbox_fit_scale():
    scale = _scale_to_bbox(mesh_bbox=[1.0, 2.0, 4.0], request_bbox=[1.0, 1.0, 1.0])

    assert scale == 0.25


def test_mesh_asset_request_scale_is_uniform_factor_not_size():
    request = {"scale": 0.25, "bbox": [1.0, 2.0, 3.0]}

    assert request_size(request) == [1.0, 2.0, 3.0]
    assert _uniform_scale_factor(request["scale"]) == 0.25


def test_planner_update_mesh_asset_metadata_action_reuses_ready_geometry(tmp_path, monkeypatch):
    case_dir = tmp_path / "case"
    mesh_path = case_dir / "assets" / "asset.obj"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text(
        "v 0 0 0\nv 2 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    manifest_path = case_dir / "assets" / "asset_manifest.json"
    original_request = {
        "name": "soft_asset",
        "asset_type": "generated_mesh",
        "purpose": "one closed soft asset",
        "scale": 1.0,
        "bbox": [2.0, 1.0, 1.0],
        "texture_needs": None,
        "simulation_role": "test role",
    }
    manifest_path.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "logical_name": "soft_asset",
                        "source_type": "generated_mesh",
                        "runtime_path": str(mesh_path),
                        "visual_path": None,
                        "scale": 1.0,
                        "bbox": [2.0, 1.0, 0.0],
                        "file_meshes_are_zup": True,
                        "texture_path": None,
                        "validation": {"manifold": {"ok": True}},
                        "asset_request": original_request,
                        "simulation_role": "test role",
                        "status": "ready",
                        "notes": [],
                    }
                ],
                "assumptions": [],
                "unresolved_risks": [],
            }
        ),
        encoding="utf-8",
    )

    def fake_validation(entry):
        return MeshGenesisFEMImportResult(
            ok=True,
            runtime_path=Path(entry["runtime_path"]),
            visual_path=None,
            scale=(0.5, 0.5, 0.5),
            file_meshes_are_zup=True,
            tet_resolution=2,
        )

    monkeypatch.setattr(mesh_episode, "run_genesis_fem_import_validation", fake_validation)
    updated_request = {**original_request, "scale": 0.5, "bbox": [1.0, 0.5, 0.5]}
    session = PlannerSession(
        PlannerSessionConfig(
            case_id="case",
            task="task",
            case_dir=case_dir,
            backend="gpu",
            timeout_sec=1.0,
            render=False,
            repair_rounds=0,
        )
    )
    action = _planner_action(
        action="update_mesh_asset_metadata",
        planner_output=_planner_output_with_asset(updated_request),
        asset_names=["soft_asset"],
    )

    result = session.actions.execute(action, turn=0)

    assert result["ok"] is True
    assert result["status"] == "mesh_asset_metadata_updated"
    assert result["metadata_updated_asset_names"] == ["soft_asset"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["assets"][0]
    assert entry["runtime_path"] == str(mesh_path)
    assert entry["scale"] == 0.5
    assert entry["status"] == "ready"
    assert entry["asset_request"] == updated_request


def _planner_output_with_asset(asset_request: dict[str, object]) -> dict[str, object]:
    return {
        "scene_brief": {
            "user_intent": "test",
            "required_entities": ["soft_asset"],
            "interaction_goal": "test",
            "success_criteria": ["test"],
            "failure_criteria": ["fail"],
            "assumptions": [],
        },
        "scene_plan": {
            "simulation_strategy": "test",
            "physics_risks": [],
            "resource_level": "low",
            "rendering_needs": [],
        },
        "asset_requests": [asset_request],
        "module_contracts": [
            {
                "owner_role": role,
                "target_files": [f"src/{role}.py"],
                "allowed_write_paths": [f"src/{role}.py"],
                "required_exports": [],
                "input_dependencies": [],
                "asset_dependencies": ["soft_asset"] if role == "body" else [],
                "forbidden_edits": [],
                "validation_expectation": None,
                "final_report_schema": "none",
            }
            for role in ("scene", "body", "action", "rendering")
        ],
        "dispatch_graph": {
            "nodes": ["scene", "body", "action", "rendering"],
            "edges": [],
            "parallel_groups": [["scene", "body", "action", "rendering"]],
            "wait_for_asset_manifest": True,
        },
        "execution_plan": {
            "mode": "local_gpu",
            "backend": "gpu",
            "duration_sec": 1.0,
            "step_budget": 1,
            "render_fps": 1,
            "render_budget": 1,
            "notes": [],
        },
        "risk_register": [],
    }


def _planner_action(
    *,
    action: str,
    planner_output: dict[str, object] | None = None,
    asset_names: list[str] | None = None,
) -> dict[str, object]:
    return {
        "action": action,
        "rationale": "test",
        "planner_output": planner_output,
        "asset_names": asset_names,
        "roles": None,
        "owner": None,
        "repair_brief": None,
        "backend": None,
        "render": None,
        "timeout_sec": None,
        "cwd": None,
        "python_args": None,
        "pytest_args": None,
        "verdict": None,
        "summary": None,
        "notes": [],
    }


def _load_generated_main(tmp_path: Path, monkeypatch, *, body_source: str | None = None):
    run_dir = tmp_path / "case"
    src_dir = run_dir / "src"
    write_main(
        run_dir=run_dir,
        task="task",
        default_steps=1,
        default_render_fps=1,
        default_duration_sec=1.0,
        default_target_video_frames=1,
        deformable_cfg={"ipc_contact_d_hat_adaptive": True},
    )
    (src_dir / "action.py").write_text("def run_actions(*args, **kwargs):\n    return None\n", encoding="utf-8")
    (src_dir / "body.py").write_text(
        body_source or "def create_bodies(*args, **kwargs):\n    return {}\n",
        encoding="utf-8",
    )
    (src_dir / "rendering.py").write_text(
        "def setup_rendering(*args, **kwargs):\n    return None\n"
        "def finalize_rendering(*args, **kwargs):\n    return None\n",
        encoding="utf-8",
    )
    (src_dir / "scene.py").write_text("def create_scene(*args, **kwargs):\n    return None\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(src_dir))
    spec = importlib.util.spec_from_file_location("generated_main_for_test", src_dir / "main.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return run_dir, module


def test_adaptive_d_hat_uses_genesis_frame_before_scale(tmp_path, monkeypatch):
    run_dir, module = _load_generated_main(tmp_path, monkeypatch)
    mesh_path = run_dir / "assets" / "triangle.obj"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 2 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    manifest = {
        "assets": [
            {
                "logical_name": "yup_triangle",
                "source_type": "input_mesh",
                "runtime_path": str(mesh_path),
                "scale": [1.0, 2.0, 3.0],
                "bbox": None,
                "file_meshes_are_zup": False,
                "simulation_role": "test mesh",
                "status": "ready",
            }
        ]
    }
    manifest_path = run_dir / "assets" / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = module._adaptive_contact_d_hat_report()

    assert report["selected_asset"] == "yup_triangle"
    assert report["selected_source_kind"] == "mesh_edges"
    assert math.isclose(report["bbox_diag"], math.sqrt(37.0), rel_tol=1e-9)
    assert math.isclose(report["median_feature_length"], 6.0, rel_tol=1e-9)
    assert math.isclose(report["global_bbox_diag"], math.sqrt(37.0), rel_tol=1e-9)
    assert math.isclose(report["selected_bbox_diag"], math.sqrt(37.0), rel_tol=1e-9)
    assert math.isclose(report["ipc_contact_d_hat"], 2e-3 * math.sqrt(37.0), rel_tol=1e-9)


def test_adaptive_d_hat_includes_bbox_only_primitive_asset(tmp_path, monkeypatch):
    run_dir, module = _load_generated_main(tmp_path, monkeypatch)
    assets_dir = run_dir / "assets"
    assets_dir.mkdir(parents=True)
    manifest = {
        "assets": [
            {
                "logical_name": "small_box_primitive",
                "source_type": "primitive",
                "runtime_path": None,
                "scale": None,
                "bbox": [0.2, 0.1, 0.1],
                "file_meshes_are_zup": None,
                "simulation_role": "bbox-only primitive collision body",
                "status": "ready",
            }
        ]
    }
    (assets_dir / "asset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = module._adaptive_contact_d_hat_report()

    expected_diag = math.sqrt(0.2**2 + 0.1**2 + 0.1**2)
    assert report["selected_asset"] == "small_box_primitive"
    assert report["selected_source_kind"] == "bbox_fallback"
    assert report["edge_count"] == 0
    assert math.isclose(report["median_feature_length"], 0.1, rel_tol=1e-9)
    assert math.isclose(report["global_bbox_diag"], expected_diag, rel_tol=1e-9)
    assert math.isclose(report["selected_bbox_diag"], expected_diag, rel_tol=1e-9)
    assert math.isclose(report["ipc_contact_d_hat"], 2e-3 * expected_diag, rel_tol=1e-9)


def test_adaptive_d_hat_uses_planner_asset_request_bbox_fallback(tmp_path, monkeypatch):
    run_dir, module = _load_generated_main(tmp_path, monkeypatch)
    contracts_dir = run_dir / "contracts"
    contracts_dir.mkdir(parents=True)
    planner_output = {
        "asset_requests": [
            {
                "name": "planned_primitive",
                "asset_type": "primitive_box",
                "scale": None,
                "bbox": [0.3, 0.2, 0.1],
                "simulation_role": "planned primitive with no mesh file",
            }
        ]
    }
    (contracts_dir / "planner_output.json").write_text(json.dumps(planner_output), encoding="utf-8")

    report = module._adaptive_contact_d_hat_report()

    expected_diag = math.sqrt(0.3**2 + 0.2**2 + 0.1**2)
    assert report["source"] == "contracts/planner_output.json asset_requests"
    assert report["selected_asset"] == "planned_primitive"
    assert report["selected_source_kind"] == "bbox_fallback"
    assert math.isclose(report["global_bbox_diag"], expected_diag, rel_tol=1e-9)
    assert math.isclose(report["ipc_contact_d_hat"], 2e-3 * expected_diag, rel_tol=1e-9)


def test_adaptive_d_hat_includes_direct_body_primitives(tmp_path, monkeypatch):
    body_source = """
TUBE_LENGTH = 2.4
TUBE_RADIUS = 0.22


def create_bodies(scene, task, *, deformable_cfg):
    scene.add_entity(
        morph=gs.morphs.Cylinder(height=TUBE_LENGTH, radius=TUBE_RADIUS, tet_resolution=2),
        material=None,
    )
    return {}
"""
    _run_dir, module = _load_generated_main(tmp_path, monkeypatch, body_source=body_source)

    report = module._adaptive_contact_d_hat_report()

    expected_diag = math.sqrt(0.44**2 + 0.44**2 + 2.4**2)
    assert report["selected_asset"] == "src/body.py:Cylinder"
    assert report["selected_source_kind"] == "direct_primitive_morph"
    assert math.isclose(report["median_feature_length"], 0.44 / 5.0, rel_tol=1e-9)
    assert math.isclose(report["global_bbox_diag"], expected_diag, rel_tol=1e-9)
    assert math.isclose(report["selected_bbox_diag"], expected_diag, rel_tol=1e-9)
    assert math.isclose(report["ipc_contact_d_hat"], 2e-3 * expected_diag, rel_tol=1e-9)


def test_adaptive_d_hat_includes_mjcf_collision_primitive_geoms(tmp_path, monkeypatch):
    run_dir, module = _load_generated_main(tmp_path, monkeypatch)
    assets_dir = run_dir / "assets"
    assets_dir.mkdir(parents=True)
    xml_path = assets_dir / "collar.xml"
    xml_path.write_text(
        """
<mujoco>
  <worldbody>
    <body name="root">
      <geom name="visual_skip" type="box" size="1 1 1" contype="0" conaffinity="0"/>
      <geom name="mount_dummy_collision" type="box" size="0.01 0.01 0.01" contype="1" conaffinity="1"/>
      <geom name="jaw_collision" type="box" size="0.085 0.025 0.060" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    manifest = {
        "assets": [
            {
                "logical_name": "collar_asset",
                "source_type": "mjcf",
                "runtime_path": str(xml_path),
                "scale": None,
                "bbox": [2.0, 1.0, 1.0],
                "file_meshes_are_zup": None,
                "simulation_role": "test collar",
                "status": "ready",
            }
        ]
    }
    (assets_dir / "asset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = module._adaptive_contact_d_hat_report()

    expected_geom_diag = math.sqrt(0.17**2 + 0.05**2 + 0.12**2)
    expected_global_diag = math.sqrt(2.0**2 + 1.0**2 + 1.0**2)
    assert report["selected_asset"] == "collar_asset/geom:jaw_collision"
    assert report["selected_source_kind"] == "mjcf_primitive_geom"
    assert math.isclose(report["median_feature_length"], 0.05, rel_tol=1e-9)
    assert math.isclose(report["selected_bbox_diag"], expected_geom_diag, rel_tol=1e-9)
    assert math.isclose(report["global_bbox_diag"], expected_global_diag, rel_tol=1e-9)
    assert math.isclose(report["ipc_contact_d_hat"], 2e-3 * expected_global_diag, rel_tol=1e-9)
