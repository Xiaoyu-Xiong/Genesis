from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import genesis as gs
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_agent.configs import RigidConfigs, rigid_config_dict
from code_agent.planner.session import PlannerSession, PlannerSessionConfig
from code_agent.utils.rigid_options import assert_scene_rigid_options, build_rigid_options
from code_agent.writer.dispatcher import _scene_rigid_options_contract_violations


def test_rigid_config_covers_all_genesis_rigid_options_and_matches_defaults():
    config_fields = {field.name for field in fields(RigidConfigs)}
    option_fields = set(gs.options.RigidOptions.model_fields)

    assert config_fields == option_fields - {"dt", "gravity"}
    assert build_rigid_options(rigid_config_dict()) == gs.options.RigidOptions()


def test_rigid_config_maps_enum_names_and_rejects_unknown_fields():
    options = build_rigid_options(
        {
            **rigid_config_dict(),
            "integrator": "Euler",
            "constraint_solver": "CG",
            "broadphase_traversal": "SAP",
        }
    )

    assert options.integrator == gs.integrator.Euler
    assert options.constraint_solver == gs.constraint_solver.CG
    assert options.broadphase_traversal == gs.broadphase_traversal.SAP
    with pytest.raises(ValueError, match="Unknown rigid config fields"):
        build_rigid_options({"not_a_rigid_option": True})


def test_scene_rigid_options_assertion_checks_resolved_sim_inheritance():
    expected = build_rigid_options()
    sim_options = gs.options.SimOptions(dt=0.02, substeps=2)
    valid_scene = SimpleNamespace(
        sim_options=sim_options,
        rigid_options=expected.model_copy_from(sim_options),
    )
    assert_scene_rigid_options(valid_scene, expected)

    invalid_scene = SimpleNamespace(
        sim_options=sim_options,
        rigid_options=gs.options.RigidOptions(iterations=51).model_copy_from(sim_options),
    )
    with pytest.raises(RuntimeError, match="overrode forced rigid options"):
        assert_scene_rigid_options(invalid_scene, expected)


def test_scene_worker_contract_requires_direct_rigid_options_forwarding():
    valid = """
def create_scene(backend, *, sim_dt, sim_substeps, rigid_options, deformable_cfg):
    return gs.Scene(rigid_options=rigid_options)
"""
    invalid = """
def create_scene(backend, *, sim_dt, sim_substeps, deformable_cfg):
    return gs.Scene(rigid_options=gs.options.RigidOptions())
"""

    assert _scene_rigid_options_contract_violations(valid) == []
    assert _scene_rigid_options_contract_violations(invalid)


def test_planner_session_writes_rigid_config_contract(tmp_path):
    session = PlannerSession(
        PlannerSessionConfig(
            case_id="rigid_config_contract",
            task="rigid sphere",
            case_dir=tmp_path / "case",
            backend="gpu",
            timeout_sec=1.0,
            render=False,
            repair_rounds=0,
        )
    )

    session._ensure_dirs()

    assert session.rigid_config_path.is_file()
    assert session.rigid_config == rigid_config_dict()
    assert json.loads(session.rigid_config_path.read_text(encoding="utf-8")) == rigid_config_dict()
