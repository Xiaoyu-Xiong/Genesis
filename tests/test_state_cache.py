from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_agent.utils.render_replay import apply_cached_state_npz
from code_agent.utils.state_cache import StateCacheError, StateCacheWriter, verify_state_cache_manifest


class _FakeEntity:
    def __init__(self, pos=(1.0, 2.0, 3.0), quat=(1.0, 0.0, 0.0, 0.0)):
        self._pos = np.asarray(pos, dtype=float)
        self._quat = np.asarray(quat, dtype=float)

    def get_pos(self):
        return self._pos

    def get_quat(self):
        return self._quat

    def set_pos(self, pos):
        self._pos = np.asarray(pos, dtype=float)

    def set_quat(self, quat):
        self._quat = np.asarray(quat, dtype=float)


class _FakeArticulatedEntity(_FakeEntity):
    n_dofs = 2
    n_links = 3

    def __init__(self, qpos=(0.1, -0.2)):
        super().__init__()
        self.qpos = np.asarray(qpos, dtype=float)
        self.dofs_position = self.qpos.copy()

    def get_qpos(self):
        return self.qpos.copy()

    def get_dofs_position(self):
        return self.dofs_position.copy()

    def set_qpos(self, qpos, *, zero_velocity=False):
        self.qpos = np.asarray(qpos, dtype=float)
        self.dofs_position = self.qpos.copy()

    def set_dofs_position(self, position, *, zero_velocity=False):
        self.dofs_position = np.asarray(position, dtype=float)
        self.qpos = self.dofs_position.copy()


class _BrokenArticulatedEntity(_FakeEntity):
    n_dofs = 1
    n_links = 2

    def set_qpos(self, qpos, *, zero_velocity=False):
        pass


def test_state_cache_writer_creates_required_npz_frames(tmp_path: Path):
    out_dir = tmp_path / "artifacts"
    actors = [{"name": "rigid_ball", "entity": _FakeEntity()}]
    writer = StateCacheWriter.create(
        out_dir=out_dir,
        scene=object(),
        actors=actors,
        steps=4,
        render_state={"capture_step_set": {0, 2, 4}},
        sim_dt=0.01,
        sim_substeps=10,
        backend="gpu",
        render_profile="debug_raster",
    )

    for step in range(5):
        writer.capture(step)
    manifest_path = writer.finalize()

    manifest = verify_state_cache_manifest(
        manifest_path,
        require_npz=True,
        require_complete_actor_state=True,
    )
    assert manifest["schema_version"] == 2
    assert manifest["npz_required"] is True
    assert manifest["state_completeness_required"] is True
    assert manifest["actor_contracts"][0]["replay_mode"] == "pose"
    assert manifest["frame_steps"] == [0, 2, 4]
    assert len(manifest["frames"]) == 3
    for frame in manifest["frames"]:
        npz_path = manifest_path.parent / frame["npz"]
        assert npz_path.is_file()
        with np.load(npz_path, allow_pickle=False) as data:
            assert data["actor_names"].tolist() == ["rigid_ball"]
            assert data["positions"].shape == (1, 3)


def test_articulated_qpos_and_dofs_are_saved_and_replayed(tmp_path: Path):
    out_dir = tmp_path / "artifacts"
    source = _FakeArticulatedEntity()
    actors = [{"name": "roller_rig", "type": "external_articulation", "entity": source}]
    writer = StateCacheWriter.create(
        out_dir=out_dir,
        scene=object(),
        actors=actors,
        steps=1,
        render_state={"capture_step_set": {0, 1}},
    )

    writer.capture(0)
    source.qpos = np.asarray([1.25, -0.75])
    source.dofs_position = source.qpos.copy()
    second_npz = writer.capture(1)
    manifest_path = writer.finalize()
    manifest = verify_state_cache_manifest(
        manifest_path,
        require_npz=True,
        require_complete_actor_state=True,
    )

    contract = manifest["actor_contracts"][0]
    assert contract["state_kind"] == "articulated"
    assert contract["replay_mode"] == "qpos"
    assert contract["required_arrays"] == ["actor_000_qpos", "actor_000_dofs_position"]
    with np.load(second_npz, allow_pickle=False) as data:
        assert np.allclose(data["actor_000_qpos"], [1.25, -0.75])
        assert np.allclose(data["actor_000_dofs_position"], [1.25, -0.75])

    target = _FakeArticulatedEntity(qpos=(0.0, 0.0))
    apply_cached_state_npz(
        second_npz,
        [{"name": "roller_rig", "entity": target}],
        actor_contracts=manifest["actor_contracts"],
    )
    assert np.allclose(target.qpos, [1.25, -0.75])


def test_state_cache_writer_rejects_dof_actor_without_generalized_state(tmp_path: Path):
    writer = StateCacheWriter.create(
        out_dir=tmp_path / "artifacts",
        scene=object(),
        actors=[{"name": "broken_rig", "entity": _BrokenArticulatedEntity()}],
        steps=0,
    )

    with pytest.raises(StateCacheError, match="no replayable qpos/DOF-position state"):
        writer.capture(0)


def test_complete_actor_contract_rejects_missing_articulated_qpos(tmp_path: Path):
    cache_dir = tmp_path / "state_cache"
    states_dir = cache_dir / "states"
    states_dir.mkdir(parents=True)
    np.savez_compressed(
        states_dir / "frame_000.npz",
        schema_version=np.asarray([2], dtype=np.int32),
        step=np.asarray([0], dtype=np.int64),
        actor_names=np.asarray(["roller_rig"], dtype="U128"),
        positions=np.asarray([[0.0, 0.0, 0.0]]),
        quats=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
    )
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "state_completeness_required": True,
                "actor_names": ["roller_rig"],
                "frames": [{"index": 0, "step": 0, "npz": "states/frame_000.npz"}],
                "actor_contracts": [
                    {
                        "index": 0,
                        "name": "roller_rig",
                        "replay_mode": "qpos",
                        "required_arrays": ["actor_000_qpos"],
                        "array_shapes": {"actor_000_qpos": [1]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(StateCacheError, match="lacks required actor array 'actor_000_qpos'"):
        verify_state_cache_manifest(
            manifest_path,
            require_npz=True,
            require_complete_actor_state=True,
        )


def test_state_cache_manifest_requires_existing_npz(tmp_path: Path):
    cache_dir = tmp_path / "state_cache"
    cache_dir.mkdir()
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frames": [{"index": 0, "step": 0, "npz": "states/frame_000.npz"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(StateCacheError, match="npz not found"):
        verify_state_cache_manifest(manifest_path, require_npz=True)


def test_state_cache_manifest_requires_npz_field(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "frames": [{"index": 0, "step": 0}]}),
        encoding="utf-8",
    )

    with pytest.raises(StateCacheError, match="missing required npz"):
        verify_state_cache_manifest(manifest_path, require_npz=True)
