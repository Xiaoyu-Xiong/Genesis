from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_agent.context.simdebug import build_simdebug_catalog, select_simdebug_cards
from code_agent.planner.session import PlannerSession, PlannerSessionConfig


def _selected_ids(selection: dict[str, object]) -> set[str]:
    cards = selection.get("selected_cards")
    assert isinstance(cards, list)
    return {str(item["id"]) for item in cards if isinstance(item, dict)}


def test_simdebug_catalog_loads_prompt_derived_cards():
    catalog = build_simdebug_catalog()

    assert catalog["library_dir"] == "code_agent/context/simdebug"
    assert catalog["card_count"] >= 25
    ids = {card["id"] for card in catalog["cards"]}
    assert "ipc_fem_material_selection_guideline" in ids
    assert "physical_causality_restriction" in ids
    assert "opt_metric_and_objective_design_guideline" in ids
    assert "planner_asset_retry_guideline" in ids
    assert "fem_ipc_initial_geometry_restriction" in ids
    assert all("provenance" in card for card in catalog["cards"])
    assert all("/cards/" not in card["source_path"] for card in catalog["cards"])


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
    assert "score" not in selection["selected_cards"][0]
    assert "matched_terms" not in selection["selected_cards"][0]
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
            deformable_enabled=True,
            ipc_enabled=True,
        )
    )
    session.contracts_dir.mkdir(parents=True)
    session.reports_dir.mkdir(parents=True)
    session.logs_dir.mkdir(parents=True)
    session.write_deformable_config_contract()

    prompt = session.planner._planner_prompt(turn=0)

    audit_path = session.reports_dir / "simdebug_card_dispatch.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    ids = _selected_ids(audit["selection"])
    assert "Planner-dispatched human debugging experience cards" in prompt
    assert "ipc_fem_material_selection_guideline" in prompt
    assert "ipc_fem_material_selection_guideline" in ids
    assert session.state["simdebug"]["latest_selected_card_ids"]
