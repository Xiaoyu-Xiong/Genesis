from __future__ import annotations

import numpy as np
import trimesh


def strip_texture_visuals(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Drop heavy material/texture objects from a mesh used only for topology work."""

    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
    return mesh


def connected_face_component_count(mesh: trimesh.Trimesh, *, face_cap: int = 100000) -> int:
    """Count face-connected components without constructing textured submeshes."""

    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) > face_cap:
        return -1
    if len(faces) == 0:
        return 0

    parent = np.arange(len(faces), dtype=np.int64)
    rank = np.zeros(len(faces), dtype=np.int8)

    def find(item: int) -> int:
        root = item
        while parent[root] != root:
            root = int(parent[root])
        while parent[item] != item:
            next_item = int(parent[item])
            parent[item] = root
            item = next_item
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            parent[left_root] = right_root
            return
        if rank[left_root] > rank[right_root]:
            parent[right_root] = left_root
            return
        parent[right_root] = left_root
        rank[left_root] += 1

    face_ids = np.repeat(np.arange(len(faces), dtype=np.int64), faces.shape[1])
    vertex_ids = faces.reshape(-1)
    order = np.argsort(vertex_ids, kind="mergesort")
    sorted_vertices = vertex_ids[order]
    sorted_faces = face_ids[order]

    start = 0
    while start < len(sorted_vertices):
        end = start + 1
        while end < len(sorted_vertices) and sorted_vertices[end] == sorted_vertices[start]:
            end += 1
        first_face = int(sorted_faces[start])
        for face in sorted_faces[start + 1 : end]:
            union(first_face, int(face))
        start = end

    roots = {find(index) for index in range(len(faces))}
    return max(1, len(roots))
