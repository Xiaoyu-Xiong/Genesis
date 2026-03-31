from __future__ import annotations

from typing import Any


def resolve_links_idx(
    entity: Any,
    *,
    links_idx_local: tuple[int, ...] | None,
    link_names: tuple[str, ...] | None,
) -> tuple[int, ...] | None:
    if links_idx_local is not None:
        return tuple(int(entity.link_start + index) for index in links_idx_local)

    if link_names is None:
        return tuple(range(int(entity.link_start), int(entity.link_end)))

    resolved: list[int] = []
    for link_name in link_names:
        link = entity.get_link(name=link_name)
        resolved.append(int(link.idx))
    return tuple(resolved)


def resolve_dofs_idx_local(
    entity: Any,
    *,
    dofs_idx_local: tuple[int, ...] | None,
    joint_names: tuple[str, ...] | None,
    values_length: int | None = None,
) -> tuple[int, ...] | None:
    if dofs_idx_local is not None:
        if values_length is not None and len(dofs_idx_local) != values_length:
            raise ValueError(
                f"Length mismatch: selected {len(dofs_idx_local)} dofs by `dofs_idx_local`, "
                f"but received {values_length} values."
            )
        return dofs_idx_local

    if joint_names is None:
        if values_length is None:
            return None
        expected = int(entity.n_dofs)
        if values_length != expected:
            raise ValueError(
                "Length mismatch without DoF selector: "
                f"entity has {expected} DoFs but received {values_length} values. "
                "Provide `dofs_idx_local` or `joint_names`, or pass full-length values."
            )
        return tuple(range(expected))

    resolved: list[int] = []
    for joint_name in joint_names:
        joint = entity.get_joint(name=joint_name)
        joint_dofs = tuple(int(index) for index in joint.dofs_idx_local)
        if len(joint_dofs) == 0:
            raise ValueError(f"Joint `{joint_name}` has no DoFs.")
        resolved.extend(joint_dofs)

    if values_length is not None and len(resolved) != values_length:
        raise ValueError(
            f"Length mismatch for joint selector: resolved {len(resolved)} dofs from `joint_names`, "
            f"but received {values_length} values."
        )

    return tuple(resolved)


def to_scalar_or_tuple(value: float | tuple[float, ...], *, expected_size: int, field_name: str) -> float | tuple[float, ...]:
    if isinstance(value, tuple):
        if len(value) != expected_size:
            raise ValueError(f"Length mismatch: `{field_name}` expects {expected_size} values, but got {len(value)}.")
        return tuple(float(component) for component in value)
    return float(value)
