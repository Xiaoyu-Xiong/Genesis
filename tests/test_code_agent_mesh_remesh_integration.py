from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import trimesh

from code_agent.assets.mesh import episode as mesh_episode
from code_agent.assets.mesh import remesh_integration
from code_agent.assets.mesh.remesh_integration import (
    apply_automatic_remesh_to_entry,
    remesh_mesh_assets_for_episode,
)
from code_agent.planner.session import PlannerSession, PlannerSessionConfig


def test_automatic_remesh_switches_manifest_paths_only_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, runtime_path = _ready_entry(tmp_path)
    entry["status"] = "failed"
    monkeypatch.setattr(remesh_integration, "_source_face_count", lambda _: 7501)
    monkeypatch.setattr(remesh_integration, "remesh_mesh_asset", _successful_fake_remesh)

    updated, outcome = apply_automatic_remesh_to_entry(
        entry,
        bundle=SimpleNamespace(texture=None, repair=SimpleNamespace(centroid_before_translation=None)),
    )

    assert outcome["status"] == "applied"
    assert outcome["applied"] is True
    assert updated["runtime_path"] != str(runtime_path.resolve())
    assert updated["remesh"]["base_runtime_path"] == str(runtime_path.resolve())
    assert updated["remesh"]["target_face_count"] == 5000
    assert updated["remesh"]["status"] == "applied"
    assert updated["validation"]["remesh"]["ok"] is True
    assert updated["status"] == "ready"


def test_automatic_remesh_skips_source_below_configured_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, runtime_path = _ready_entry(tmp_path)

    def unexpected_remesh(_config):
        raise AssertionError("The remesher must not run for a source already below the configured target.")

    monkeypatch.setattr(remesh_integration, "remesh_mesh_asset", unexpected_remesh)

    updated, outcome = apply_automatic_remesh_to_entry(
        entry,
        bundle=SimpleNamespace(texture=None, repair=SimpleNamespace(centroid_before_translation=None)),
    )

    assert outcome == {
        "ok": True,
        "status": "skipped_not_needed",
        "applied": False,
        "fallback_used": False,
        "source_face_count": 320,
        "target_face_count": 5000,
        "target_face_tolerance": 0.5,
        "skip_face_count_upper_bound": 7500,
    }
    assert updated["runtime_path"] == str(runtime_path.resolve())
    assert updated["remesh"]["attempted"] is False
    assert updated["remesh"]["source_face_count"] == 320
    assert not (runtime_path.parent.parent / "remesh").exists()


def test_automatic_remesh_skips_source_within_configured_tolerance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, runtime_path = _ready_entry(tmp_path)
    monkeypatch.setattr(remesh_integration, "_source_face_count", lambda _: 7500)

    def unexpected_remesh(_config):
        raise AssertionError("The remesher must not run inside the configured upper tolerance.")

    monkeypatch.setattr(remesh_integration, "remesh_mesh_asset", unexpected_remesh)

    updated, outcome = apply_automatic_remesh_to_entry(
        entry,
        bundle=SimpleNamespace(texture=None, repair=SimpleNamespace(centroid_before_translation=None)),
    )

    assert outcome["status"] == "skipped_not_needed"
    assert outcome["source_face_count"] == 7500
    assert outcome["skip_face_count_upper_bound"] == 7500
    assert updated["runtime_path"] == str(runtime_path.resolve())
    assert updated["remesh"]["attempted"] is False
    assert updated["remesh"]["target_face_tolerance"] == 0.5
    assert updated["remesh"]["skip_face_count_upper_bound"] == 7500


def test_mesh_episode_runs_automatic_remesh_before_original_genesis_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pre_validation_entry = {
        "logical_name": "dense_mesh",
        "source_type": "generated_mesh",
        "runtime_path": "/tmp/dense.obj",
        "status": "failed",
        "validation": {"manifold": {"ok": True}},
    }
    remeshed_entry = {**pre_validation_entry, "status": "ready", "remesh": {"status": "applied"}}
    bundle = SimpleNamespace(to_dict=lambda: {})
    monkeypatch.setattr(mesh_episode, "process_downloaded_meshy_mesh", lambda **_: bundle)
    monkeypatch.setattr(mesh_episode, "manifest_entry_from_bundle", lambda *_: pre_validation_entry)
    monkeypatch.setattr(
        mesh_episode,
        "apply_automatic_remesh_to_entry",
        lambda *_args, **_kwargs: (
            remeshed_entry,
            {"ok": True, "status": "applied", "applied": True, "fallback_used": False},
        ),
    )

    def unexpected_original_import(*_args, **_kwargs):
        raise AssertionError("Original high-density Genesis import must be skipped after automatic remesh succeeds.")

    monkeypatch.setattr(mesh_episode, "run_genesis_fem_import_validation", unexpected_original_import)

    result = mesh_episode._process_one_mesh_asset(
        {
            "ok": True,
            "request": {"name": "dense_mesh", "asset_type": "generated_mesh"},
            "mesh_prompt": "dense test mesh",
            "downloaded": object(),
        }
    )

    assert result["ok"] is True
    assert result["manifest_entry"]["remesh"]["status"] == "applied"
    assert result["auto_remesh"]["applied"] is True


def test_automatic_remesh_failure_falls_back_to_original_ready_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, runtime_path = _ready_entry(tmp_path)
    monkeypatch.setattr(remesh_integration, "_source_face_count", lambda _: 12000)
    monkeypatch.setattr(remesh_integration, "remesh_mesh_asset", _failed_fake_remesh)

    updated, outcome = apply_automatic_remesh_to_entry(
        entry,
        bundle=SimpleNamespace(texture=None, repair=SimpleNamespace(centroid_before_translation=None)),
    )

    assert outcome["status"] == "failed_fallback_original"
    assert outcome["fallback_used"] is True
    assert updated["runtime_path"] == str(runtime_path.resolve())
    assert updated["visual_path"] == str(runtime_path.resolve())
    assert updated["status"] == "ready"
    assert updated["remesh"]["applied"] is False
    assert updated["remesh"]["fallback_used"] is True


def test_planner_remesh_action_is_exposed_and_commits_validated_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_dir = tmp_path / "case"
    entry, runtime_path = _ready_entry(case_dir)
    entry["status"] = "failed"
    _write_manifest(case_dir, entry)
    monkeypatch.setattr(remesh_integration, "_source_face_count", lambda _: 12000)
    monkeypatch.setattr(remesh_integration, "remesh_mesh_asset", _successful_fake_remesh)
    session = PlannerSession(
        PlannerSessionConfig(
            case_id="case",
            task="test planner remesh",
            case_dir=case_dir,
            backend="gpu",
            timeout_sec=1.0,
            render=False,
            repair_rounds=0,
        )
    )
    action = _planner_action(target_face_count=4000)

    assert not session.validate_json_schema(action, Path("code_agent/specs/planner_action.schema.json"))
    result = session.actions.execute(action, turn=0)

    assert result["ok"] is True
    assert result["status"] == "mesh_assets_remeshed"
    assert result["remeshed_asset_names"] == ["mesh_asset"]
    manifest = json.loads((case_dir / "assets" / "asset_manifest.json").read_text(encoding="utf-8"))
    updated = manifest["assets"][0]
    assert updated["runtime_path"] != str(runtime_path.resolve())
    assert updated["remesh"]["mode"] == "planner"
    assert updated["remesh"]["base_runtime_path"] == str(runtime_path.resolve())


def test_planner_remesh_failure_keeps_manifest_and_ready_asset_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_dir = tmp_path / "case"
    entry, _ = _ready_entry(case_dir)
    _write_manifest(case_dir, entry)
    manifest_path = case_dir / "assets" / "asset_manifest.json"
    before = manifest_path.read_text(encoding="utf-8")
    monkeypatch.setattr(remesh_integration, "_source_face_count", lambda _: 12000)
    monkeypatch.setattr(remesh_integration, "remesh_mesh_asset", _failed_fake_remesh)
    session = PlannerSession(
        PlannerSessionConfig(
            case_id="case",
            task="test failed planner remesh",
            case_dir=case_dir,
            backend="gpu",
            timeout_sec=1.0,
            render=False,
            repair_rounds=0,
        )
    )

    result = session.actions.execute(_planner_action(target_face_count=4000), turn=0)

    assert result["ok"] is False
    assert result["status"] == "mesh_asset_remesh_failed"
    assert manifest_path.read_text(encoding="utf-8") == before
    assert session.state["assets"]["jobs"]["mesh"]["ok"] is True
    assert session.state["assets"]["jobs"]["mesh"]["last_remesh_ok"] is False


def test_planner_remesh_runtime_requires_exactly_one_target(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    entry, _ = _ready_entry(case_dir)
    _write_manifest(case_dir, entry)
    session = PlannerSession(
        PlannerSessionConfig(
            case_id="case",
            task="test schema",
            case_dir=case_dir,
            backend="gpu",
            timeout_sec=1.0,
            render=False,
            repair_rounds=0,
        )
    )

    no_target = _planner_action()
    two_targets = _planner_action(target_face_count=4000, target_edge_length=0.01)
    edge_target = _planner_action(target_edge_length=0.01)

    assert not session.validate_json_schema(no_target, Path("code_agent/specs/planner_action.schema.json"))
    assert not session.validate_json_schema(two_targets, Path("code_agent/specs/planner_action.schema.json"))
    assert not session.validate_json_schema(edge_target, Path("code_agent/specs/planner_action.schema.json"))
    assert session.actions.execute(no_target, turn=0)["status"] == "precondition_failed"
    assert session.actions.execute(two_targets, turn=1)["status"] == "precondition_failed"


def _ready_entry(root: Path) -> tuple[dict[str, object], Path]:
    asset_root = root / "assets" / "mesh" / "00_mesh_asset"
    runtime_path = asset_root / "processed" / "repaired.obj"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.icosphere(subdivisions=2).export(runtime_path)
    return {
        "logical_name": "mesh_asset",
        "source_type": "generated_mesh",
        "runtime_path": str(runtime_path.resolve()),
        "visual_path": str(runtime_path.resolve()),
        "scale": 1.0,
        "bbox": [2.0, 2.0, 2.0],
        "file_meshes_are_zup": False,
        "texture_path": None,
        "validation": {"manifold": {"ok": True}, "genesis_fem_import": {"ok": True}},
        "asset_request": {"name": "mesh_asset", "asset_type": "generated_mesh"},
        "simulation_role": "test generated mesh",
        "status": "ready",
        "notes": [],
    }, runtime_path


def _write_manifest(case_dir: Path, entry: dict[str, object]) -> None:
    path = case_dir / "assets" / "asset_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"assets": [entry], "assumptions": [], "unresolved_risks": []}, indent=2) + "\n",
        encoding="utf-8",
    )


def _planner_action(
    *, target_face_count: int | None = None, target_edge_length: float | None = None
) -> dict[str, object]:
    return {
        "action": "remesh_mesh_assets",
        "rationale": "The generated mesh is valid but too dense.",
        "planner_output": None,
        "asset_names": ["mesh_asset"],
        "target_face_count": target_face_count,
        "target_edge_length": target_edge_length,
        "target_face_tolerance": None,
        "roles": None,
        "owner": None,
        "repair_brief": None,
        "backend": None,
        "render": None,
        "render_profile": None,
        "save_state_cache": None,
        "require_state_cache": None,
        "replay_cache": None,
        "render_only": None,
        "timeout_sec": None,
        "cwd": None,
        "python_args": None,
        "pytest_args": None,
        "verdict": None,
        "summary": None,
        "notes": [],
        "simdebug_cards": None,
    }


def _successful_fake_remesh(config) -> dict[str, object]:
    processed = config.output_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    runtime_path = processed / "repaired.obj"
    trimesh.creation.icosphere(subdivisions=1).export(runtime_path)
    return {
        "ok": True,
        "standalone": True,
        "pipeline_integrated": False,
        "source": {"face_count": 12000},
        "request": {
            "target_face_count": config.target_face_count,
            "target_edge_length": config.target_edge_length,
            "target_face_tolerance": config.target_face_tolerance,
        },
        "target_check": {"ok": True},
        "output": {
            "face_count": 4200,
            "bbox_min": [-1.0, -1.0, -1.0],
            "bbox_max": [1.0, 1.0, 1.0],
        },
        "manifold_validation": {"ok": True, "tetgen_ready": True},
        "texture_validation": {"requested": False, "ok": True},
        "genesis_fem_import_validation": {
            "ok": True,
            "rigid_import": {"ok": True},
            "volumetric_fem_import": {"ok": True},
            "cloth_import": {"ok": True},
        },
        "artifacts": {
            "runtime_mesh": str(runtime_path),
            "visual_mesh": None,
            "base_color_texture": None,
            "report": str(config.output_dir / "remesh_report.json"),
        },
    }


def _failed_fake_remesh(config) -> dict[str, object]:
    return {
        "ok": False,
        "standalone": True,
        "pipeline_integrated": False,
        "failure_stage": "manifold_validation",
        "source": {"face_count": 12000},
        "request": {
            "target_face_count": config.target_face_count,
            "target_edge_length": config.target_edge_length,
            "target_face_tolerance": config.target_face_tolerance,
        },
        "output": {"face_count": 3000},
        "manifold_validation": {"ok": False, "error": "synthetic manifold failure"},
        "artifacts": {
            "runtime_mesh": str(config.output_dir / "processed" / "repaired.obj"),
            "visual_mesh": None,
            "base_color_texture": None,
            "report": str(config.output_dir / "remesh_report.json"),
        },
    }
