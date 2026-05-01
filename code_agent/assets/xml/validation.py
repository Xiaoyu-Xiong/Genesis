from __future__ import annotations

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from code_agent.assets.xml.validation_core.collectors import (
    collect_actuators,
    collect_body_tree,
    collect_equalities,
    collect_tendons,
    forbidden_elements,
    global_simulation_elements,
    has_mesh_asset,
    single_child,
    tag,
)
from code_agent.assets.xml.validation_core.manifest import manifest_entry_from_xml_validation
from code_agent.assets.xml.validation_core.rules import (
    base_contract,
    control_interface,
    validate_actuator_contract,
    validate_geoms,
    validate_joint_contract,
)


def validate_xml_asset(xml_path: Path) -> dict[str, Any]:
    """Validate a generated MJCF asset without running a Genesis simulation."""

    xml_path = xml_path.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    report = _empty_report(xml_path, errors, warnings)

    if not xml_path.exists():
        errors.append(f"XML file does not exist: {xml_path}")
        return report

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        errors.append(f"XML parser error: {exc}")
        return report

    report["parser_ok"] = True
    root = tree.getroot()
    if tag(root) != "mujoco":
        errors.append(f"Root element must be <mujoco>, found <{tag(root)}>.")

    _validate_asset_scope(root, errors)
    worldbody = _validate_worldbody(root, errors)
    direct_root_bodies = _direct_root_bodies(worldbody)

    body_infos: list[dict[str, Any]] = []
    joint_infos: list[dict[str, Any]] = []
    geom_infos: list[dict[str, Any]] = []
    site_infos: list[dict[str, Any]] = []
    if direct_root_bodies:
        collect_body_tree(
            direct_root_bodies[0],
            parent_path="",
            body_infos=body_infos,
            joint_infos=joint_infos,
            geom_infos=geom_infos,
            site_infos=site_infos,
            errors=errors,
            warnings=warnings,
        )

    tendons = collect_tendons(root)
    actuators = collect_actuators(root)
    equalities = collect_equalities(root)
    report.update(
        {
            "bodies": body_infos,
            "joints": joint_infos,
            "geoms": geom_infos,
            "sites": site_infos,
            "tendons": tendons,
            "actuators": actuators,
            "equalities": equalities,
            "base": base_contract(direct_root_bodies[0] if direct_root_bodies else None),
            "control_interface": control_interface(actuators, equalities),
        }
    )

    validate_joint_contract(joint_infos, errors, warnings)
    validate_actuator_contract(
        actuators=actuators,
        joint_infos=joint_infos,
        tendons=tendons,
        site_infos=site_infos,
        body_infos=body_infos,
        equalities=equalities,
        errors=errors,
        warnings=warnings,
    )
    validate_geoms(geom_infos, errors, warnings)
    _run_mujoco_import(xml_path, report, errors)

    report["ok"] = report["parser_ok"] and report["mujoco_ok"] and not errors
    return report


def _empty_report(xml_path: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    return {
        "ok": False,
        "xml_path": str(xml_path),
        "parser_ok": False,
        "mujoco_ok": False,
        "model_summary": {},
        "errors": errors,
        "warnings": warnings,
        "joints": [],
        "actuators": [],
        "sites": [],
        "tendons": [],
        "equalities": [],
        "bodies": [],
        "geoms": [],
        "base": {},
        "control_interface": {},
    }


def _validate_asset_scope(root: ET.Element, errors: list[str]) -> None:
    forbidden = forbidden_elements(root)
    if forbidden:
        errors.append("Forbidden scene-level or external elements found: " + ", ".join(forbidden))

    global_settings = global_simulation_elements(root)
    if global_settings:
        errors.append(
            "Global simulation settings do not belong in standalone XML assets: " + ", ".join(global_settings)
        )

    if has_mesh_asset(root):
        errors.append("Generated XML assets must not use mesh or hfield assets; use primitive MJCF geoms.")


def _validate_worldbody(root: ET.Element, errors: list[str]) -> ET.Element | None:
    worldbody = single_child(root, "worldbody", errors)
    if worldbody is None:
        return None

    direct_root_bodies = _direct_root_bodies(worldbody)
    direct_non_body_tags = [tag(child) for child in list(worldbody) if tag(child) != "body"]
    if len(direct_root_bodies) != 1:
        errors.append(
            f"Exactly one direct articulated body tree is required under <worldbody>; "
            f"found {len(direct_root_bodies)}."
        )
    if direct_non_body_tags:
        errors.append(
            "No scene-level objects are allowed directly under <worldbody>; found: "
            + ", ".join(direct_non_body_tags)
        )
    return worldbody


def _direct_root_bodies(worldbody: ET.Element | None) -> list[ET.Element]:
    if worldbody is None:
        return []
    return [child for child in list(worldbody) if tag(child) == "body"]


def _run_mujoco_import(xml_path: Path, report: dict[str, Any], errors: list[str]) -> None:
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(xml_path))
    except Exception as exc:  # noqa: BLE001 - MuJoCo parser errors should be preserved in the report.
        errors.append(f"MuJoCo import error: {type(exc).__name__}: {exc}")
        return

    report["mujoco_ok"] = True
    report["model_summary"] = {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "ngeom": int(model.ngeom),
        "nsite": int(model.nsite),
        "ntendon": int(model.ntendon),
        "stat_center": [float(value) for value in model.stat.center],
        "stat_extent": float(model.stat.extent),
    }
