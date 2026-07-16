from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_agent.evaluation.agent import _normalize_cache_source_consistency, _source_change_worker_evidence_paths


def _mismatch_report():
    return {
        "status": "mismatch",
        "manifest_path": "/case/artifacts/state_cache/manifest.json",
        "mismatches": [{"path": "src/scene.py"}],
    }


def test_critic_render_only_classification_can_reuse_mismatched_cache():
    normalized = _normalize_cache_source_consistency(
        {
            "classification": "render_only",
            "mismatched_files": ["src/scene.py"],
            "physics_rerun_required": True,
            "rationale": "Only RayTracer environment parameters changed.",
            "evidence": ["reports/state_cache_source_diffs/src__scene.py.diff"],
        },
        _mismatch_report(),
    )

    assert normalized["classification"] == "render_only"
    assert normalized["physics_rerun_required"] is False


def test_critic_physics_classification_always_requires_rerun():
    normalized = _normalize_cache_source_consistency(
        {
            "classification": "physics_affecting",
            "mismatched_files": ["src/scene.py"],
            "physics_rerun_required": False,
            "rationale": "The rigid solver timestep changed.",
            "evidence": ["reports/state_cache_source_diffs/src__scene.py.diff"],
        },
        _mismatch_report(),
    )

    assert normalized["classification"] == "physics_affecting"
    assert normalized["physics_rerun_required"] is True


def test_critic_cannot_call_a_deterministic_mismatch_a_match():
    normalized = _normalize_cache_source_consistency(
        {
            "classification": "match",
            "mismatched_files": [],
            "physics_rerun_required": False,
            "rationale": "No change.",
            "evidence": [],
        },
        _mismatch_report(),
    )

    assert normalized["classification"] == "indeterminate"
    assert normalized["physics_rerun_required"] is True


def test_critic_schema_requires_cache_source_consistency():
    schema_path = Path(__file__).resolve().parents[1] / "code_agent" / "specs" / "critic_report.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert "cache_source_consistency" in schema["required"]
    classification = schema["properties"]["cache_source_consistency"]["properties"]["classification"]
    assert set(classification["enum"]) == {
        "not_applicable",
        "match",
        "render_only",
        "physics_affecting",
        "indeterminate",
    }


def test_source_mismatch_points_critic_to_matching_worker_reports(tmp_path: Path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    scene_report = logs_dir / "codex_scene_repair.final.json"
    scene_report.write_text('{"changed_files":["src/scene.py"]}', encoding="utf-8")
    (logs_dir / "codex_action_repair.final.json").write_text("{}", encoding="utf-8")

    paths = _source_change_worker_evidence_paths(tmp_path, _mismatch_report())

    assert paths == [str(scene_report.resolve())]
