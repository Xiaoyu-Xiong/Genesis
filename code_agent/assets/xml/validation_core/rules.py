from __future__ import annotations

import xml.etree.ElementTree as ET

from code_agent.assets.xml.validation_core.collectors import tag


def validate_joint_contract(
    joint_infos: list[dict[str, object]], errors: list[str], warnings: list[str]
) -> None:
    if not joint_infos:
        errors.append("XML asset must contain at least one named joint.")
        return
    non_free_joints = [joint for joint in joint_infos if joint.get("type") != "free"]
    if not non_free_joints:
        errors.append("XML asset must contain at least one controllable non-free articulated joint.")
    for joint in non_free_joints:
        joint_type = joint.get("type")
        if joint_type in {"hinge", "slide"} and not joint.get("range"):
            warnings.append(f"Joint `{joint['name']}` has no explicit range.")


def validate_actuator_contract(
    *,
    actuators: list[dict[str, object]],
    joint_infos: list[dict[str, object]],
    tendons: list[dict[str, object]],
    site_infos: list[dict[str, object]],
    body_infos: list[dict[str, object]],
    equalities: list[dict[str, object]],
    errors: list[str],
    warnings: list[str],
) -> None:
    if not actuators:
        errors.append("XML asset must include at least one actuator for its articulated body.")
        return

    joint_names = {str(joint.get("name")) for joint in joint_infos}
    tendon_names = {str(tendon.get("name")) for tendon in tendons}
    site_names = {str(site.get("name")) for site in site_infos}
    body_names = {str(body.get("name")) for body in body_infos}
    non_free_joints = {str(joint.get("name")) for joint in joint_infos if joint.get("type") != "free"}
    targeted_non_free_joints: set[str] = set()

    for actuator in actuators:
        name = actuator.get("name")
        if str(name).startswith("unnamed_"):
            errors.append("Every actuator must be named.")
        targets = {
            "joint": actuator.get("joint"),
            "tendon": actuator.get("tendon"),
            "site": actuator.get("site"),
            "body": actuator.get("body"),
        }
        populated = {key: value for key, value in targets.items() if value}
        if len(populated) != 1:
            errors.append(f"Actuator `{name}` must target exactly one joint, tendon, site, or body.")
            continue
        key, target = next(iter(populated.items()))
        if key == "joint":
            if target not in joint_names:
                errors.append(f"Actuator `{name}` targets unknown joint `{target}`.")
            elif target in non_free_joints:
                targeted_non_free_joints.add(str(target))
        elif key == "tendon" and target not in tendon_names:
            errors.append(f"Actuator `{name}` targets unknown tendon `{target}`.")
        elif key == "site" and target not in site_names:
            errors.append(f"Actuator `{name}` targets unknown site `{target}`.")
        elif key == "body" and target not in body_names:
            errors.append(f"Actuator `{name}` targets unknown body `{target}`.")

        if not actuator.get("ctrlrange"):
            warnings.append(f"Actuator `{name}` has no explicit ctrlrange; neutral command defaults to 0.")

    equality_coupled_joints = equality_coupled_to_targets(equalities, targeted_non_free_joints)
    missing = sorted(non_free_joints - targeted_non_free_joints - equality_coupled_joints)
    if missing:
        warnings.append("Non-free joints without direct actuator coverage: " + ", ".join(missing))


def validate_geoms(geom_infos: list[dict[str, object]], errors: list[str], warnings: list[str]) -> None:
    if not geom_infos:
        errors.append("XML asset must include primitive geoms for its articulated body.")
        return
    collision_geoms = [
        geom
        for geom in geom_infos
        if geom.get("contype") not in {"0", "0.0"} and geom.get("conaffinity") not in {"0", "0.0"}
    ]
    if not collision_geoms:
        errors.append("XML asset must include at least one collision-enabled primitive geom.")
    untyped = [str(geom["name"]) for geom in geom_infos if not geom.get("type")]
    if untyped:
        warnings.append("Some geoms rely on default geom type: " + ", ".join(untyped))


def base_contract(root_body: ET.Element | None) -> dict[str, object]:
    if root_body is None:
        return {"mode": "unknown", "placement_expectation": "validation failed before root body detection"}
    has_freejoint = any(tag(child) == "freejoint" for child in list(root_body))
    has_free_type_joint = any(
        tag(child) == "joint" and child.attrib.get("type") == "free" for child in list(root_body)
    )
    if has_freejoint or has_free_type_joint:
        return {
            "mode": "free",
            "placement_expectation": "Scene/Body workers should place the root body as a movable free-base asset.",
        }
    return {
        "mode": "fixed",
        "placement_expectation": "Scene placement fixes the MJCF root body to the world unless Genesis adds a free base.",
    }


def control_interface(
    actuators: list[dict[str, object]], equalities: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "actuators": actuators,
        "suggested_commands": [
            {
                "name": actuator["name"],
                "neutral_command": actuator.get("neutral_command", 0.0),
                "command_range": actuator.get("ctrlrange"),
                "schedule_hint": schedule_hint(actuator),
            }
            for actuator in actuators
        ],
        "coupling": equality_coupling_notes(equalities),
        "notes": [
            "Action code should command these named actuators directly instead of inferring control semantics from raw XML."
        ],
    }


def schedule_hint(actuator: dict[str, object]) -> str:
    actuator_type = actuator.get("type")
    if actuator_type == "position":
        return "Use smooth setpoint ramps within ctrlrange."
    if actuator_type == "velocity":
        return "Use short velocity pulses, then return to neutral."
    if actuator_type == "motor":
        return "Use torque/force pulses with neutral hold at 0."
    return "Use neutral hold and bounded open-loop commands within ctrlrange."


def equality_coupled_to_targets(equalities: list[dict[str, object]], targeted_joints: set[str]) -> set[str]:
    coupled: set[str] = set()
    for equality in equalities:
        if equality.get("type") != "joint":
            continue
        joint1 = equality.get("joint1")
        joint2 = equality.get("joint2")
        if not joint1 or not joint2:
            continue
        if joint1 in targeted_joints:
            coupled.add(str(joint2))
        if joint2 in targeted_joints:
            coupled.add(str(joint1))
    return coupled


def equality_coupling_notes(equalities: list[dict[str, object]]) -> list[str]:
    notes: list[str] = []
    for equality in equalities:
        equality_type = equality.get("type")
        name = equality.get("name")
        if equality_type == "joint":
            notes.append(
                f"Equality `{name}` couples joint `{equality.get('joint1')}` to `{equality.get('joint2')}` "
                f"with polycoef={equality.get('polycoef')}."
            )
        elif equality_type:
            notes.append(f"Equality `{name}` uses type `{equality_type}`.")
    return notes
