from __future__ import annotations

from typing import Any


_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _normalized_severity(issue: dict[str, Any]) -> str | None:
    severity = issue.get("severity")
    if not isinstance(severity, str):
        return None
    severity = severity.strip().lower()
    return severity if severity in _SEVERITY_ORDER else None


def _filtered_sorted_issues(section: dict[str, Any]) -> list[dict[str, Any]]:
    issues = section.get("issues")
    if not isinstance(issues, list):
        return []

    filtered: list[dict[str, Any]] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        severity = _normalized_severity(issue)
        if severity == "low":
            continue
        filtered.append(issue)

    filtered.sort(key=lambda issue: _SEVERITY_ORDER.get(_normalized_severity(issue) or "medium", 99))
    return filtered


def _issue_targets(section_name: str, issue: dict[str, Any]) -> tuple[str, ...]:
    if section_name == "scene":
        return ("ir",)
    if section_name == "actions":
        return ("actions",)

    title = issue.get("title")
    fix = issue.get("fix")
    evidence = issue.get("evidence")
    parts: list[str] = []
    if isinstance(title, str):
        parts.append(title)
    if isinstance(fix, str):
        parts.append(fix)
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, str):
                parts.append(item)
    text = " ".join(parts).lower()

    ir_markers = (
        "body.initial_pose",
        "bodies[",
        "bodies[].",
        "initial_pose",
        "scene.render",
        "follow_entity",
        "camera_pos",
        "camera_fov",
        "ir ",
        "in the ir",
        "actions list",
    )
    xml_markers = (
        "xml",
        "mjcf",
        "regenerate",
        "joint",
        "damping",
        "frictionloss",
        "support geometry",
        "support layout",
        "chassis support",
        "caster",
        "skid",
        "wheel-joint",
        "wheel joint",
        "<body",
        "root offset",
    )

    targets: list[str] = []
    if any(marker in text for marker in ir_markers):
        targets.append("ir")
    if any(marker in text for marker in xml_markers):
        targets.append("xml")
    if not targets:
        targets.append("ir")
    return tuple(dict.fromkeys(targets))


def _issue_record(section_name: str, issue: dict[str, Any]) -> dict[str, Any]:
    severity = _normalized_severity(issue) or "medium"
    return {
        "section": section_name,
        "severity": severity,
        "title": issue.get("title"),
        "fix": issue.get("fix"),
        "targets": list(_issue_targets(section_name, issue)),
    }


def _format_issue_line(issue: dict[str, Any]) -> list[str]:
    severity = issue["severity"]
    title = issue.get("title")
    fix = issue.get("fix")
    lines: list[str] = []
    if isinstance(title, str) and title.strip():
        lines.append(f"- [{severity}] {title.strip()}")
    if isinstance(fix, str) and fix.strip():
        lines.append(f"  fix [{severity}]: {fix.strip()}")
    return lines


def _build_generator_requirements(package: dict[str, Any]) -> str:
    lines = [
        "The previous candidate did not pass critic review.",
        "Keep the original task intent unchanged.",
        "Revise the generated IR (and XML if articulated) to address the issues below.",
        "Prioritize `high` severity issues before `medium` severity issues.",
        "Ignore `low` severity issues for this refinement pass.",
        "Before producing the next candidate, make an internal repair plan that maps each `high` severity issue to concrete IR or XML changes.",
        "Do not regenerate a broadly similar candidate and hope it works; apply targeted repairs to the known blockers.",
    ]

    verdict = package.get("verdict")
    if isinstance(verdict, str) and verdict:
        lines.extend(["", f"Previous critic verdict: {verdict}"])

    summary = package.get("summary")
    if isinstance(summary, str) and summary.strip():
        lines.extend(["", "Critic summary:", summary.strip()])

    for section_name in ("scene", "actions"):
        section = package["section_feedback"].get(section_name, {})
        section_summary = section.get("summary")
        issues = section.get("issues", [])
        lines.extend(["", f"{section_name.capitalize()} feedback:"])
        if isinstance(section_summary, str) and section_summary.strip():
            lines.append(section_summary.strip())
        for issue in issues:
            lines.extend(_format_issue_line(issue))

    for body_name, section in package.get("body_feedback", {}).items():
        section_summary = section.get("summary")
        issues = section.get("issues", [])
        lines.extend(["", f"Body `{body_name}` feedback:"])
        if isinstance(section_summary, str) and section_summary.strip():
            lines.append(section_summary.strip())
        for issue in issues:
            lines.extend(_format_issue_line(issue))

    must_fix = package.get("must_fix", [])
    if must_fix:
        lines.extend(["", "Must-fix items for this round:"])
        for item in must_fix:
            lines.append(f"- {item}")

    xml_requirements_by_body = package.get("xml_requirements_by_body")
    if isinstance(xml_requirements_by_body, dict) and xml_requirements_by_body:
        lines.extend(
            [
                "",
                "If you call `generate_articulated_xml`, you must pass XML-specific repair requirements through that tool call.",
                "The XML-specific repair requirements are provided separately per articulated body and must not be ignored.",
            ]
        )

    lines.extend(
        [
            "",
            "Requirements for the next attempt:",
            "- Return a revised candidate that directly addresses the critic issues.",
            "- Fix `high` severity issues first, then `medium` severity issues if capacity remains.",
            "- Preserve already-good parts unless a higher-priority fix requires changing them.",
            "- Do not keep known-bad geometry, timing, or control choices if the critic flagged them.",
            "- Preserve rendering and observation support.",
        ]
    )
    return "\n".join(lines)


def _build_xml_requirements(xml_issues: list[dict[str, Any]]) -> str | None:
    if not xml_issues:
        return None

    lines = [
        "XML-specific repair requirements for this round:",
        "Apply these changes to the articulated XML itself, not only to IR actions.",
        "Prioritize `high` severity XML issues first.",
    ]
    for issue in xml_issues:
        lines.extend(_format_issue_line(issue))
    return "\n".join(lines)


def build_generator_feedback_package(critic_analysis: dict[str, Any]) -> dict[str, Any]:
    by_section = critic_analysis.get("by_section")
    if not isinstance(by_section, dict):
        by_section = {}
    by_body = critic_analysis.get("by_body")
    if not isinstance(by_body, dict):
        by_body = {}

    section_feedback: dict[str, Any] = {}
    body_feedback: dict[str, Any] = {}
    must_fix: list[str] = []
    xml_issue_records_by_body: dict[str, list[dict[str, Any]]] = {}

    for section_name in ("scene", "actions"):
        section = by_section.get(section_name)
        if not isinstance(section, dict):
            section = {}
        issues = [_issue_record(section_name, issue) for issue in _filtered_sorted_issues(section)]
        section_feedback[section_name] = {
            "summary": section.get("summary"),
            "issues": issues,
        }

        for issue in issues:
            severity = issue["severity"]
            title = issue.get("title")
            fix = issue.get("fix")
            issue_text = f"[{severity}] {title}: {fix}" if isinstance(title, str) and isinstance(fix, str) else None
            if severity == "high" and isinstance(issue_text, str):
                must_fix.append(issue_text)

            if not isinstance(fix, str) or not fix.strip():
                continue

    for body_name, section in by_body.items():
        if not isinstance(body_name, str) or not isinstance(section, dict):
            continue
        issues = [_issue_record("body", issue) for issue in _filtered_sorted_issues(section)]
        body_feedback[body_name] = {
            "summary": section.get("summary"),
            "issues": issues,
        }

        for issue in issues:
            severity = issue["severity"]
            title = issue.get("title")
            fix = issue.get("fix")
            issue_text = (
                f"[{severity}] {body_name}: {title}: {fix}"
                if isinstance(title, str) and isinstance(fix, str)
                else None
            )
            if severity == "high" and isinstance(issue_text, str):
                must_fix.append(issue_text)

            if not isinstance(fix, str) or not fix.strip():
                continue
            if "xml" in issue["targets"]:
                xml_issue_records_by_body.setdefault(body_name, []).append(issue)

    package: dict[str, Any] = {
        "verdict": critic_analysis.get("verdict"),
        "summary": critic_analysis.get("summary"),
        "section_feedback": section_feedback,
        "body_feedback": body_feedback,
        "must_fix": must_fix,
    }
    package["xml_requirements_by_body"] = {
        body_name: text
        for body_name, issues in sorted(xml_issue_records_by_body.items())
        if (text := _build_xml_requirements(issues)) is not None
    }
    package["generator_requirements"] = _build_generator_requirements(package)
    return package
