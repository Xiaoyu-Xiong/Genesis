from __future__ import annotations

from collections.abc import Mapping

import genesis as gs

from code_agent.configs import rigid_config_dict


_ENUM_FIELDS = {
    "integrator": gs.integrator,
    "constraint_solver": gs.constraint_solver,
    "broadphase_traversal": gs.broadphase_traversal,
}


def build_rigid_options(config: Mapping[str, object] | None = None) -> gs.options.RigidOptions:
    """Build RigidOptions from the complete repository-owned config contract."""

    data = rigid_config_dict()
    if config is not None:
        unknown = set(config).difference(data)
        if unknown:
            raise ValueError(f"Unknown rigid config fields: {sorted(unknown)}")
        data.update(config)
    for name, enum_type in _ENUM_FIELDS.items():
        value = data.get(name)
        if isinstance(value, str):
            try:
                data[name] = enum_type[value]
            except KeyError as exc:
                raise ValueError(f"Invalid rigid config value {name}={value!r}") from exc
    return gs.options.RigidOptions(**data)


def assert_scene_rigid_options(scene, expected: gs.options.RigidOptions) -> None:
    """Reject generated scenes that did not apply the forced rigid options."""

    actual = getattr(scene, "rigid_options", None)
    sim_options = getattr(scene, "sim_options", None)
    if actual is None or sim_options is None:
        raise RuntimeError("Generated create_scene() did not return a Genesis scene with rigid options")
    resolved_expected = expected.model_copy_from(sim_options)
    if actual != resolved_expected:
        actual_data = actual.model_dump(mode="python")
        expected_data = resolved_expected.model_dump(mode="python")
        mismatches = {
            name: {"expected": expected_data.get(name), "actual": actual_data.get(name)}
            for name in expected_data
            if actual_data.get(name) != expected_data.get(name)
        }
        raise RuntimeError(f"Generated scene overrode forced rigid options: {mismatches}")
