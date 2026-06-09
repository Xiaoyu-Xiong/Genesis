from __future__ import annotations

from typing import Any

from code_agent.configs import CONFIGS
from code_agent.writer.common import WorkerRole
from code_agent.writer.dispatcher import (
    dispatch_worker_roles,
    repair_worker,
    resolve_writer_parallelism,
    write_worker_dispatch_report,
)

WORKER_ROLES: tuple[WorkerRole, ...] = ("scene", "body", "action", "rendering")


class WorkerActionHandler:
    """Planner action handlers for code-writing workers and targeted repair."""

    def __init__(self, session: Any):
        self.session = session

    def spawn_workers(self, action: dict[str, Any]) -> dict[str, Any]:
        planner_output = self.session.current_planner_output()
        if planner_output is None:
            return {"ok": False, "status": "precondition_failed", "message": "planner_output is missing."}
        blocked_roles = roles_requiring_asset_manifest(planner_output, roles=self.roles_from_action(action))
        if blocked_roles and not self.session.asset_manifest_ready():
            assets = self.session.state.get("assets", {})
            status = assets.get("status") if isinstance(assets, dict) else None
            return {
                "ok": False,
                "status": "precondition_failed",
                "message": (
                    "These roles require a ready asset manifest before they run: "
                    f"{', '.join(blocked_roles)}. Current asset status: {status or 'unknown'}."
                ),
                "roles_requiring_asset_manifest": list(blocked_roles),
            }
        roles = self.roles_from_action(action)
        if not roles:
            return {"ok": False, "status": "invalid_action", "message": "spawn_workers requires at least one role."}
        simdebug_card_context_by_role = {
            role: self.session.simdebug_card_context_for_role(
                role,
                turn=self.session.state.get("turn_index"),
                dispatch_reason="spawn_workers",
                requested_card_ids=self.session.simdebug_card_ids_from_action(action, role),
                extra_state={"planner_action": action, "roles": list(roles)},
            )
            for role in roles
        }
        results = dispatch_worker_roles(
            case_dir=self.session.case_dir,
            task=self.session.config.task,
            planner_output=planner_output,
            roles=roles,
            repair_context=action.get("repair_brief") if isinstance(action.get("repair_brief"), str) else None,
            simdebug_card_context_by_role=simdebug_card_context_by_role,
        )
        self.session.record_worker_results(results)
        write_worker_dispatch_report(self.session.case_dir, results)
        all_ok = self.session.all_workers_ok()
        if all_ok:
            self.session.state["control"]["needs_integration"] = True
            self.session.state["control"]["needs_execution"] = False
            self.session.state["control"]["needs_critic"] = False
        active_parallelism = resolve_writer_parallelism(len(roles))
        return {
            "ok": all(item.ok for item in results),
            "status": "workers_dispatched",
            "roles": list(roles),
            "parallel": active_parallelism > 1,
            "max_parallel_workers": active_parallelism,
            "configured_max_parallel_workers": CONFIGS.harness.max_parallel_workers,
            "all_workers_ok": all_ok,
        }

    def request_repair(self, action: dict[str, Any]) -> dict[str, Any]:
        planner_output = self.session.current_planner_output()
        if planner_output is None:
            return {"ok": False, "status": "precondition_failed", "message": "planner_output is missing."}
        budgets = self.session.state["budgets"]
        if int(budgets["repair_attempts"]) >= int(budgets["max_repair_rounds"]):
            return {"ok": False, "status": "budget_exhausted", "message": "repair budget exhausted."}
        owner = str(action.get("owner") or self.session.recommended_owner())
        repair_brief = action.get("repair_brief")
        if not isinstance(repair_brief, str) or not repair_brief.strip():
            repair_brief = self.session.failure_context()
        simdebug_card_context = self.session.simdebug_card_context_for_role(
            owner,
            turn=self.session.state.get("turn_index"),
            dispatch_reason="request_repair",
            requested_card_ids=self.session.simdebug_card_ids_from_action(action, owner),
            extra_state={"planner_action": action, "repair_brief": repair_brief},
        )
        repaired = repair_worker(
            case_dir=self.session.case_dir,
            task=self.session.config.task,
            owner=owner,
            failure_context=repair_brief,
            simdebug_card_context=simdebug_card_context,
        )
        budgets["repair_attempts"] = int(budgets["repair_attempts"]) + 1
        if repaired is None:
            return {"ok": False, "status": "invalid_owner", "message": f"Cannot repair owner {owner!r}."}
        self.session.record_worker_results([repaired])
        write_worker_dispatch_report(self.session.case_dir, [repaired])
        self.session.state["control"]["needs_integration"] = self.session.all_workers_ok()
        self.session.state["control"]["needs_execution"] = False
        self.session.state["control"]["needs_critic"] = False
        return {
            "ok": repaired.ok,
            "status": "repair_dispatched",
            "owner": owner,
            "all_workers_ok": self.session.all_workers_ok(),
        }

    def roles_from_action(self, action: dict[str, Any]) -> tuple[WorkerRole, ...]:
        raw_roles = action.get("roles")
        if not isinstance(raw_roles, list):
            return ()
        roles: list[WorkerRole] = []
        for role in raw_roles:
            if role in WORKER_ROLES and role not in roles:
                roles.append(role)
        return tuple(roles)


def roles_requiring_asset_manifest(
    planner_output: dict[str, Any] | None,
    *,
    roles: tuple[WorkerRole, ...],
) -> tuple[WorkerRole, ...]:
    if not isinstance(planner_output, dict) or not roles:
        return ()
    contracts = planner_output.get("module_contracts")
    if not isinstance(contracts, list):
        return ()
    blocked: list[WorkerRole] = []
    role_set = set(roles)
    for contract in contracts:
        if not isinstance(contract, dict):
            continue
        role = contract.get("owner_role")
        if role not in role_set:
            continue
        asset_dependencies = contract.get("asset_dependencies")
        input_dependencies = contract.get("input_dependencies")
        has_asset_dependencies = isinstance(asset_dependencies, list) and bool(asset_dependencies)
        has_manifest_input = (
            isinstance(input_dependencies, list)
            and any(
                isinstance(item, str) and ("asset_manifest" in item or "assets/asset_manifest" in item)
                for item in input_dependencies
            )
        )
        if has_asset_dependencies or has_manifest_input:
            blocked.append(role)
    return tuple(blocked)
