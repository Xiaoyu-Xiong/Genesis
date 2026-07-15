from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_agent.context.simdebug import build_simdebug_catalog, select_simdebug_cards
from code_agent.configs import deformable_config_dict
from code_agent.opt.types import OptAgentRequest
from code_agent.planner.session import PlannerSession, PlannerSessionConfig
from code_agent.prompts.opt import build_opt_prompt
from code_agent.writer.dispatcher import WORKERS, _worker_prompt


def _selected_ids(selection: dict[str, object]) -> set[str]:
    cards = selection.get("selected_cards")
    assert isinstance(cards, list)
    return {str(item["id"]) for item in cards if isinstance(item, dict)}


def test_simdebug_catalog_loads_prompt_derived_cards():
    catalog = build_simdebug_catalog()

    assert catalog["library_dir"] == "code_agent/context/simdebug"
    assert catalog["card_count"] >= 36
    ids = {card["id"] for card in catalog["cards"]}
    assert "ipc_fem_material_selection_guideline" in ids
    assert "physical_causality_restriction" in ids
    assert "opt_metric_and_objective_design_guideline" in ids
    assert "planner_asset_retry_guideline" in ids
    assert "fem_ipc_initial_geometry_restriction" in ids
    assert "planner_card_dispatch_guideline" in ids
    assert "rigid_contact_metrics_guideline" in ids
    assert "two_stage_rendering_workflow_guideline" in ids
    assert "final_path_tracing_siggraph_guideline" in ids
    assert "physics_state_cache_replay_guideline" in ids
    assert "render_replay_consistency_guideline" in ids
    assert "controller_schedule_guideline" in ids
    assert "actuation_stress_relief_guideline" in ids
    assert "soft_body_robust_layout_guideline" in ids
    assert "fem_cloth_shell_guideline" in ids
    assert "asset_inspection_decision_guideline" in ids
    assert "ipc_world_invalid_failure_signature_guideline" in ids
    assert all("provenance" in card for card in catalog["cards"])
    assert all("kind" not in card for card in catalog["cards"])
    source_paths = [str(card["source_path"]) for card in catalog["cards"]]
    assert all(path.startswith("code_agent/context/simdebug/") for path in source_paths)
    assert all("/cards/" not in path for path in source_paths)
    assert all("/guideline_cards/" not in path and "/restriction_cards/" not in path for path in source_paths)
    category_dirs = {Path(path).parts[3] for path in source_paths}
    assert category_dirs == {
        "assets_geometry",
        "contact_collision",
        "control_dynamics",
        "deformable_fem",
        "diagnosis_repair",
        "evidence_validation",
        "optimization",
        "rendering_quality",
        "workflow_orchestration",
    }


def test_render_background_cards_distinguish_debug_and_final_profiles():
    catalog = build_simdebug_catalog()
    cards = {card["id"]: card for card in catalog["cards"]}

    evidence_text = json.dumps(cards["render_visual_evidence_restriction"], ensure_ascii=False)
    final_text = json.dumps(cards["final_path_tracing_siggraph_guideline"], ensure_ascii=False)
    workflow_text = json.dumps(cards["two_stage_rendering_workflow_guideline"], ensure_ascii=False)
    ground_text = json.dumps(cards["non_white_ground_on_white_background_guideline"], ensure_ascii=False)

    assert "Pure white is a fallback for fast inspection, not a hard requirement" in evidence_text
    assert "final_path_traced or replay_render" in evidence_text
    assert "do not inherit the debug evidence pure-white fallback" in evidence_text
    assert "Prefer light studio backdrops" in final_text
    assert "soft off-white" in final_text
    assert "Do not inherit the debug raster pure-white background fallback" in final_text
    assert "background_style" in final_text
    assert "path_tracing.enabled=false" in final_text
    assert "iterative look-dev" in final_text
    assert "RGBA alpha" in final_text
    assert "soft contact shadows" in final_text
    assert "RayTracer sphere lights as renderable geometry" in final_text
    assert "light_visibility_checks" in final_text
    assert "white background does not waive this check" in final_text
    assert "Glass(color=(1.0, 1.0, 1.0)" in final_text
    assert "Render-only replay must not add, hide, delete, or replace geometry" in final_text
    assert "single Glass sphere renders as a solid glass volume" in final_text
    assert "screen, window pane, or transparent plate" in final_text
    assert "alpha_rgba_used=false" in final_text
    assert "replay_only=true" in final_text
    assert "does not mean the case can finish" in workflow_text
    assert "first RayTracer output" in workflow_text
    assert "This card does not require a white background" in ground_text


def test_camera_palette_geometry_and_cache_cards_include_hard_replay_checks():
    cards = {card["id"]: card for card in build_simdebug_catalog()["cards"]}
    camera_text = json.dumps(cards["camera_framing_subject_fill_guideline"], ensure_ascii=False)
    style_text = json.dumps(cards["visual_style_readability_guideline"], ensure_ascii=False)
    geometry_text = json.dumps(cards["generated_asset_shape_fidelity_guideline"], ensure_ascii=False)
    cache_text = json.dumps(cards["physics_state_cache_replay_guideline"], ensure_ascii=False)
    replay_text = json.dumps(cards["render_replay_consistency_guideline"], ensure_ascii=False)

    assert "whole-body center of mass" in camera_text
    assert "Do not follow a selected vertex" in camera_text
    assert "Do not apply unsmoothed per-frame COM" in camera_text
    assert "two consecutive final-render attempts" in style_text
    assert "Preserve prompt-specified colors" in style_text
    assert "bounded thin layer" in geometry_text
    assert "Articulated rigid bodies must save" in cache_text
    assert "Root pose alone is insufficient" in cache_text
    assert "actor contract" in cache_text
    assert "articulated actor lacks qpos/DOF state" in replay_text


def test_simdebug_selector_picks_fem_material_cards_for_soft_ipc_task():
    selection = select_simdebug_cards(
        {
            "task": "soft FEM IPC dual column bridge leveling with visible compression and tuned material friction",
            "deformable_enabled": True,
            "ipc_enabled": True,
        },
        target_role="planner",
    )

    ids = _selected_ids(selection)
    assert "ipc_fem_material_selection_guideline" in ids
    assert "deformable_fem_ipc_scope_restriction" in ids
    assert "soft_body_robust_layout_guideline" in ids
    assert "actuation_stress_relief_guideline" in ids
    assert "planner_card_dispatch_guideline" in ids
    assert "score" not in selection["selected_cards"][0]
    assert "matched_terms" not in selection["selected_cards"][0]
    assert "kind" not in selection["selected_cards"][0]
    assert selection["physics_modes"] == ["fem_ipc"]


def test_simdebug_selector_picks_fem_cloth_card_for_cloth_task():
    selection = select_simdebug_cards(
        {
            "task": "FEM cloth sheet drapes over a rigid cylinder with IPC contact",
            "deformable_enabled": True,
            "ipc_enabled": True,
        },
        target_role="body",
    )

    ids = _selected_ids(selection)
    assert "fem_cloth_shell_guideline" in ids
    assert "deformable_fem_ipc_scope_restriction" in ids
    assert selection["physics_modes"] == ["fem_ipc"]


def test_simdebug_selector_exposes_role_and_physics_candidates_for_planner_judgment():
    selection = select_simdebug_cards(
        {
            "task": "rigid IPC chain links should interlock and settle on the table",
            "ipc_enabled": True,
            "critic": {
                "latest_error": (
                    "World is not valid after SimplicialSurfaceIntersectionCheck; "
                    "later exception mentions rigid ABD state retrieval."
                )
            },
        },
        target_role="critic",
    )

    ids = _selected_ids(selection)
    assert "ipc_initial_geometry_failure_diagnosis_guideline" in ids
    assert "ipc_world_invalid_failure_signature_guideline" in ids
    assert "collision_contact_restriction" in ids
    assert selection["selection_policy"] == "all_role_and_physics_compatible_candidates_for_planner_relevance_judgment"


def test_simdebug_selector_does_not_send_fem_only_card_to_rigid_task():
    selection = select_simdebug_cards(
        {
            "task": "rigid ball rolls across a table and contacts a block",
            "ipc_enabled": False,
            "deformable_enabled": False,
        },
        target_role="planner",
    )

    assert "ipc_fem_material_selection_guideline" not in _selected_ids(selection)
    assert selection["physics_modes"] == ["rigid"]


def test_simdebug_selector_accepts_legacy_kind_field_without_leaking_it():
    catalog = {
        "schema_version": 1,
        "cards": [
            {
                "schema_version": 1,
                "id": "legacy_collision_card",
                "kind": "restriction",
                "title": "Legacy collision card",
                "summary": "Old catalogs may still carry kind during migration.",
                "scopes": ["planner", "body"],
                "physics_modes": ["any"],
                "restrictions": ["Keep physical collision evidence readable."],
                "source_path": "code_agent/context/simdebug/legacy/legacy_collision_card.yaml",
            }
        ],
    }

    selection = select_simdebug_cards(
        {"task": "rigid contact", "ipc_enabled": False, "deformable_enabled": False},
        target_role="body",
        catalog=catalog,
    )

    assert _selected_ids(selection) == {"legacy_collision_card"}
    selected = selection["selected_cards"][0]
    assert "kind" not in selected
    assert "kind" not in selected["card"]


def test_source_aware_repair_card_can_be_dispatched_to_repair_workers():
    case_state = {
        "task": "rigid mechanism fails because the body structure and action timing need source-aware repair",
        "ipc_enabled": False,
        "deformable_enabled": False,
    }

    body_selection = select_simdebug_cards(
        case_state,
        target_role="body",
        requested_card_ids=("source_aware_repair_guideline",),
    )
    action_selection = select_simdebug_cards(
        case_state,
        target_role="action",
        requested_card_ids=("source_aware_repair_guideline",),
    )
    rendering_selection = select_simdebug_cards(
        case_state,
        target_role="rendering",
        requested_card_ids=("source_aware_repair_guideline",),
    )

    assert "source_aware_repair_guideline" in _selected_ids(body_selection)
    assert "source_aware_repair_guideline" in _selected_ids(action_selection)
    assert "source_aware_repair_guideline" not in _selected_ids(rendering_selection)


def test_planner_prompt_includes_simdebug_dispatch_and_writes_audit(tmp_path):
    case_dir = tmp_path / "case"
    session = PlannerSession(
        PlannerSessionConfig(
            case_id="soft_case",
            task="soft FEM IPC column bridge leveling with material friction",
            case_dir=case_dir,
            backend="gpu",
            timeout_sec=1.0,
            render=False,
            repair_rounds=0,
        )
    )
    _mark_session_fem_ipc_selected(session)
    session.contracts_dir.mkdir(parents=True)
    session.reports_dir.mkdir(parents=True)
    session.logs_dir.mkdir(parents=True)
    session.write_deformable_config_contract()

    prompt = session.planner._planner_prompt(turn=0)

    audit_path = session.reports_dir / "simdebug_card_dispatch.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    ids = _selected_ids(audit["selection"])
    assert "Planner-dispatched human debugging experience cards" in prompt
    assert "[card] ipc_fem_material_selection_guideline" in prompt
    assert "[guideline]" not in prompt
    assert "[restriction]" not in prompt
    assert "ipc_fem_material_selection_guideline" in prompt
    assert "ipc_fem_material_selection_guideline" in ids
    assert session.state["simdebug"]["latest_selected_card_ids"]


def test_planner_session_auto_physics_exposes_all_initial_modes(tmp_path):
    session = PlannerSession(
        PlannerSessionConfig(
            case_id="mixed_case",
            task="mixed rigid, cloth, or soft cases are selected per prompt",
            case_dir=tmp_path / "case",
            backend="gpu",
            timeout_sec=1.0,
            render=False,
            repair_rounds=0,
        )
    )

    assert set(session.simdebug_physics_modes()) == {"rigid", "rigid_ipc", "fem_ipc"}
    assert session.deformable_config["enabled"] is False
    assert session.deformable_config["ipc_enabled"] is False


def test_legacy_prompts_are_preserved_and_active_prompts_are_slimmed():
    repo_root = Path(__file__).resolve().parents[1]
    legacy_ipc = repo_root / "code_agent" / "prompts_legacy" / "ipc.py"
    active_ipc = repo_root / "code_agent" / "prompts" / "ipc.py"
    from code_agent.prompts_legacy.ipc import FEM_MATERIAL_SELECTION_GUIDE as legacy_fem_material_guide

    assert "1e4` to `5e4" in legacy_ipc.read_text(encoding="utf-8")
    assert "1e4` to `5e4" not in active_ipc.read_text(encoding="utf-8")
    assert "Planner-dispatched SimDebug cards" in active_ipc.read_text(encoding="utf-8")
    assert "1e4` to `5e4" in legacy_fem_material_guide


def test_prompt_mode_env_switches_to_legacy_prompts():
    repo_root = Path(__file__).resolve().parents[1]
    code = (
        "from code_agent.prompts.worker import WORKER_COMMON_RULES; "
        "from code_agent.prompts.planner import PLANNER_GENERAL_RULES; "
        "print('cards' if 'Planner-dispatched SimDebug cards' in WORKER_COMMON_RULES else 'legacy'); "
        "print('legacy_scale' if 'Scale policy:' in PLANNER_GENERAL_RULES else 'no_legacy_scale')"
    )

    active = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    legacy = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env={**dict(os.environ), "CODE_AGENT_PROMPT_MODE": "legacy"},
    )

    assert active.stdout.splitlines() == ["cards", "no_legacy_scale"]
    assert legacy.stdout.splitlines() == ["legacy", "legacy_scale"]


def test_legacy_prompt_mode_disables_simdebug_dispatch():
    repo_root = Path(__file__).resolve().parents[1]
    code = """
from pathlib import Path
from tempfile import TemporaryDirectory
from code_agent.planner.session import PlannerSession, PlannerSessionConfig
with TemporaryDirectory() as tmp:
    s = PlannerSession(PlannerSessionConfig(case_id='case', task='rigid ball rolls', case_dir=Path(tmp) / 'case', backend='gpu', timeout_sec=1.0, render=False, repair_rounds=0))
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    s.logs_dir.mkdir(parents=True, exist_ok=True)
    s.contracts_dir.mkdir(parents=True, exist_ok=True)
    print(s.simdebug_cards_enabled())
    print(bool(s.simdebug_card_context_for_role('planner')))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env={**dict(os.environ), "CODE_AGENT_PROMPT_MODE": "legacy"},
    )

    assert result.stdout.splitlines() == ["False", "False"]


def test_simdebug_cards_env_flag_disables_dispatch_without_legacy_prompts():
    repo_root = Path(__file__).resolve().parents[1]
    code = """
from pathlib import Path
from tempfile import TemporaryDirectory
from code_agent.planner.session import PlannerSession, PlannerSessionConfig
from code_agent.prompts import prompt_mode
from code_agent.prompts.worker import WORKER_COMMON_RULES
with TemporaryDirectory() as tmp:
    s = PlannerSession(PlannerSessionConfig(case_id='case', task='soft FEM IPC contact', case_dir=Path(tmp) / 'case', backend='gpu', timeout_sec=1.0, render=False, repair_rounds=0))
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    s.logs_dir.mkdir(parents=True, exist_ok=True)
    s.contracts_dir.mkdir(parents=True, exist_ok=True)
    print(prompt_mode())
    print('cards_prompt' if 'Planner-dispatched SimDebug cards' in WORKER_COMMON_RULES else 'legacy_prompt')
    print(s.simdebug_cards_enabled())
    print(bool(s.simdebug_card_context_for_role('planner')))
    print((s.reports_dir / 'simdebug_card_dispatch.json').exists())
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env={**dict(os.environ), "CODE_AGENT_SIMDEBUG_CARDS": "0", "CODE_AGENT_PROMPT_MODE": "cards"},
    )

    assert result.stdout.splitlines() == ["cards", "cards_prompt", "False", "False", "False"]


def test_session_dispatches_role_specific_simdebug_cards(tmp_path):
    session = _simdebug_test_session(tmp_path)

    body_context = session.simdebug_card_context_for_role("body", turn=1, dispatch_reason="test_body")
    opt_context = session.simdebug_card_context_for_role("opt", turn=1, dispatch_reason="test_opt")

    assert "ipc_fem_material_selection_guideline" in body_context
    assert "opt_effective_parameter_restriction" in opt_context
    assert (session.reports_dir / "simdebug_card_dispatch_body.json").exists()
    assert (session.reports_dir / "simdebug_card_dispatch_opt.json").exists()
    assert session.state["simdebug"]["latest_by_role"]["body"]["selected_card_ids"]


def test_session_honors_planner_requested_simdebug_card_ids(tmp_path):
    session = _simdebug_test_session(tmp_path)
    action = {
        "simdebug_cards": {
            "body": ["ipc_fem_material_selection_guideline"],
            "opt": ["opt_effective_parameter_restriction"],
        }
    }

    body_ids = session.simdebug_card_ids_from_action(action, "body")
    body_context = session.simdebug_card_context_for_role(
        "body",
        turn=2,
        dispatch_reason="test_requested_body",
        requested_card_ids=body_ids,
    )
    audit = json.loads((session.reports_dir / "simdebug_card_dispatch_body.json").read_text(encoding="utf-8"))

    assert body_ids == ("ipc_fem_material_selection_guideline",)
    assert "ipc_fem_material_selection_guideline" in body_context
    assert "physical_causality_restriction" not in body_context
    assert audit["requested_card_ids"] == ["ipc_fem_material_selection_guideline"]
    assert audit["selection"]["selection_policy"] == (
        "planner_requested_ids_filtered_by_declared_role_scope_and_active_physics_mode"
    )


def test_worker_and_opt_prompts_receive_planner_dispatched_cards(tmp_path):
    session = _simdebug_test_session(tmp_path)
    body_context = session.simdebug_card_context_for_role("body", turn=0, dispatch_reason="test_worker_prompt")
    worker_prompt = _worker_prompt(
        case_dir=session.case_dir,
        task=session.config.task,
        planner_output={"module_contracts": []},
        asset_manifest={"assets": []},
        deformable_config=session.deformable_config,
        genesis_context="Genesis context pointer",
        spec=WORKERS["body"],
        repair_context=None,
        simdebug_card_context=body_context,
    )
    assert "Planner-dispatched SimDebug cards for this worker" in worker_prompt
    assert "ipc_fem_material_selection_guideline" in worker_prompt

    opt_context = session.simdebug_card_context_for_role("opt", turn=0, dispatch_reason="test_opt_prompt")
    opt_prompt = build_opt_prompt(OptAgentRequest(case_dir=session.case_dir, simdebug_card_context=opt_context))
    assert "Planner-dispatched SimDebug cards for Opt" in opt_prompt
    assert "opt_effective_parameter_restriction" in opt_prompt
    assert '"simdebug_card_context"' not in opt_prompt.split("Planner-dispatched SimDebug cards for Opt:", 1)[0]


def test_ipc_contact_coupling_tuning_bounds_are_exposed():
    cfg = deformable_config_dict(physics_mode="fem_ipc")

    assert cfg["ipc_contact_resistance_default"] == 1e7
    assert cfg["ipc_contact_resistance_min"] == 3e6
    assert cfg["ipc_contact_resistance"] == 1e7
    assert cfg["ipc_contact_resistance_max"] == 1e8
    assert cfg["ipc_constraint_strength_translation_default"] == 30
    assert cfg["ipc_constraint_strength_translation_min"] == 10
    assert cfg["ipc_constraint_strength_translation"] == 30
    assert cfg["ipc_constraint_strength_translation_max"] == 100
    assert cfg["ipc_constraint_strength_rotation_default"] == 30
    assert cfg["ipc_constraint_strength_rotation_min"] == 10
    assert cfg["ipc_constraint_strength_rotation"] == 30
    assert cfg["ipc_constraint_strength_rotation_max"] == 100


def _simdebug_test_session(tmp_path) -> PlannerSession:
    case_dir = tmp_path / "case"
    session = PlannerSession(
        PlannerSessionConfig(
            case_id="soft_case",
            task="soft FEM IPC column bridge leveling with material friction",
            case_dir=case_dir,
            backend="gpu",
            timeout_sec=1.0,
            render=False,
            repair_rounds=0,
        )
    )
    _mark_session_fem_ipc_selected(session)
    session.contracts_dir.mkdir(parents=True)
    session.reports_dir.mkdir(parents=True)
    session.logs_dir.mkdir(parents=True)
    session.write_deformable_config_contract()
    return session


def _mark_session_fem_ipc_selected(session: PlannerSession) -> None:
    physics_plan = {
        "mode": "fem_ipc",
        "deformable_enabled": True,
        "deformable_kind": "soft_body",
        "ipc_enabled": True,
        "rationale": "test FEM/IPC simdebug coverage",
    }
    session.deformable_config = deformable_config_dict(physics_mode="fem_ipc")
    session._sync_capability_state(physics_plan)
