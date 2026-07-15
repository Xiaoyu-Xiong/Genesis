from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_agent.planner.action_handlers import runtime_actions
from code_agent.utils import execution
from code_agent.utils.integrator import write_main


def test_integrator_writes_state_cache_and_replay_cli(tmp_path: Path):
    main_py = write_main(
        run_dir=tmp_path / "case",
        task="rigid ball rolls",
        default_steps=4,
        default_render_fps=25,
        default_duration_sec=1.0,
        default_target_video_frames=2,
    )

    text = main_py.read_text(encoding="utf-8")
    assert "--render-profile" in text
    assert "_call_with_optional_render_profile" in text
    assert "render_profile=args.render_profile" in text
    assert "--save-state-cache" in text
    assert "--require-state-cache" in text
    assert 'parser.add_argument("--save-state-cache", action="store_true", default=True)' in text
    assert 'parser.add_argument("--require-state-cache", action="store_true", default=True)' in text
    assert "--replay-cache" in text
    assert "--render-only" in text
    assert "verify_state_cache_manifest(" in text
    assert "require_complete_actor_state=True" in text
    assert "StateCacheWriter.create" in text
    assert "run_render_only_replay" in text
    assert 'stats["render_profile"] = args.render_profile' in text


def test_execution_debug_raster_mode_uses_default_environment(tmp_path: Path, monkeypatch):
    captured = _capture_run_local(monkeypatch)

    execution.run_generated_simulation(
        main_py=tmp_path / "case" / "src" / "main.py",
        run_dir=tmp_path / "case",
        backend="gpu",
        timeout_sec=1.0,
        steps=2,
        render_fps=25,
        sim_dt=0.01,
        sim_substeps=1,
        render_every_n_steps=1,
        render_res=(64, 48),
    )

    config = captured[0]
    assert ("--render-profile", "debug_raster") == _arg_pair(config.extra_args, "--render-profile")
    assert "--save-state-cache" in config.extra_args
    assert "--require-state-cache" in config.extra_args
    assert "GENESIS_PATH_TRACING_OPTIX_DIR" not in config.env
    assert config.env["GENESIS_RENDER_PROFILE"] == "debug_raster"


def test_execution_final_path_traced_mode_sets_optix_environment(tmp_path: Path, monkeypatch):
    captured = _capture_run_local(monkeypatch)

    execution.run_generated_simulation(
        main_py=tmp_path / "case" / "src" / "main.py",
        run_dir=tmp_path / "case",
        backend="gpu",
        timeout_sec=1.0,
        steps=2,
        render_fps=25,
        sim_dt=0.01,
        sim_substeps=1,
        render_every_n_steps=1,
        render_res=(64, 48),
        render_profile="final_path_traced",
        save_state_cache=True,
        require_state_cache=True,
    )

    config = captured[0]
    assert ("--render-profile", "final_path_traced") == _arg_pair(config.extra_args, "--render-profile")
    assert "--save-state-cache" in config.extra_args
    assert "--require-state-cache" in config.extra_args
    assert config.env["GENESIS_RENDER_PROFILE"] == "final_path_traced"
    assert config.env["GENESIS_PATH_TRACING_OPTIX_DIR"] == "/opt/nvidia-optix-595/lib"
    assert config.env["LD_LIBRARY_PATH"].split(":")[0] == "/opt/nvidia-optix-595/lib"


def test_execution_render_only_replay_mode_sets_cache_args_and_optix_environment(tmp_path: Path, monkeypatch):
    captured = _capture_run_local(monkeypatch)
    replay_cache = tmp_path / "cache" / "manifest.json"

    execution.run_generated_simulation(
        main_py=tmp_path / "case" / "src" / "main.py",
        run_dir=tmp_path / "case",
        backend="gpu",
        timeout_sec=1.0,
        steps=2,
        render_fps=25,
        sim_dt=0.01,
        sim_substeps=1,
        render_every_n_steps=1,
        render_res=(64, 48),
        replay_cache=replay_cache,
        render_only=True,
    )

    config = captured[0]
    assert ("--replay-cache", str(replay_cache)) == _arg_pair(config.extra_args, "--replay-cache")
    assert "--render-only" in config.extra_args
    assert "--save-state-cache" not in config.extra_args
    assert "--require-state-cache" not in config.extra_args
    assert config.env["GENESIS_PATH_TRACING_OPTIX_DIR"] == "/opt/nvidia-optix-595/lib"
    assert config.env["LD_LIBRARY_PATH"].split(":")[0] == "/opt/nvidia-optix-595/lib"


def test_runtime_action_forces_state_cache_for_physics_execution(tmp_path: Path, monkeypatch):
    case_dir = tmp_path / "case"
    (case_dir / "src").mkdir(parents=True)
    (case_dir / "src" / "main.py").write_text("# generated\n", encoding="utf-8")
    captured = {}

    def fake_run_generated_simulation(**kwargs):
        captured.update(kwargs)

        class Report:
            ok = True

            def to_dict(self):
                return {"ok": True, "command": ["uv"], "returncode": 0}

        return Report()

    monkeypatch.setattr(runtime_actions, "run_generated_simulation", fake_run_generated_simulation)
    session = _runtime_session(case_dir)
    handler = runtime_actions.RuntimeActionHandler(session)

    result = handler.run_execution(
        {
            "backend": None,
            "render": None,
            "render_profile": "debug_raster",
            "save_state_cache": False,
            "require_state_cache": False,
            "replay_cache": None,
            "render_only": False,
        }
    )

    assert result["ok"] is True
    assert captured["render_profile"] == "debug_raster"
    assert captured["render_only"] is False
    assert captured["save_state_cache"] is True
    assert captured["require_state_cache"] is True
    assert session.state["execution"]["save_state_cache"] is True
    assert session.state["execution"]["require_state_cache"] is True


def test_runtime_action_passes_final_path_tracing_replay_options(tmp_path: Path, monkeypatch):
    case_dir = tmp_path / "case"
    (case_dir / "src").mkdir(parents=True)
    main_py = case_dir / "src" / "main.py"
    main_py.write_text("# generated\n", encoding="utf-8")
    captured = {}

    def fake_run_generated_simulation(**kwargs):
        captured.update(kwargs)

        class Report:
            ok = True

            def to_dict(self):
                return {"ok": True, "command": ["uv"], "returncode": 0}

        return Report()

    monkeypatch.setattr(runtime_actions, "run_generated_simulation", fake_run_generated_simulation)
    session = _runtime_session(case_dir)
    handler = runtime_actions.RuntimeActionHandler(session)

    result = handler.run_execution(
        {
            "backend": None,
            "render": None,
            "render_profile": "final_path_traced",
            "save_state_cache": None,
            "require_state_cache": None,
            "replay_cache": "artifacts/state_cache/manifest.json",
            "render_only": True,
        }
    )

    assert result["ok"] is True
    assert captured["render_profile"] == "final_path_traced"
    assert captured["replay_cache"] == case_dir / "artifacts/state_cache/manifest.json"
    assert captured["render_only"] is True
    assert captured["save_state_cache"] is False
    assert captured["require_state_cache"] is False
    assert session.state["execution"]["render_profile"] == "final_path_traced"
    assert session.state["final_render"]["attempts"] == 1


def test_finish_pass_requires_final_path_traced_acceptance(tmp_path: Path):
    session = _runtime_session(tmp_path / "case")
    session.state["critic"] = {"verdict": "pass"}
    session.state["final_render"] = {"required": True, "status": "needed"}
    handler = runtime_actions.RuntimeActionHandler(session)

    result = handler.finish({"verdict": "pass", "summary": "done", "rationale": "done"})

    assert result["ok"] is False
    assert result["status"] == "precondition_failed"
    assert "final_path_traced" in result["message"]


def _capture_run_local(monkeypatch):
    captured = []

    def fake_run_local(config):
        captured.append(config)
        return {
            "command": ["uv", "run", "--no-sync", "python", config.main_file, *config.extra_args],
            "exit_code": 0,
            "duration_sec": 0.01,
            "stdout_path": str(config.output_dir / "stdout.txt"),
            "stderr_path": str(config.output_dir / "stderr.txt"),
            "artifact_paths": [],
            "diagnostics": {},
        }

    monkeypatch.setattr(execution, "run_local", fake_run_local)
    return captured


def _arg_pair(args: tuple[str, ...], flag: str) -> tuple[str, str]:
    index = args.index(flag)
    return args[index], args[index + 1]


def _runtime_session(case_dir: Path):
    class Session:
        def __init__(self):
            self.case_dir = case_dir
            self.config = SimpleNamespace(backend="gpu", render=True, timeout_sec=1.0)
            self.state = {
                "integration": {"main_py": str(case_dir / "src" / "main.py")},
                "execution": None,
                "critic": None,
                "control": {"needs_execution": True, "needs_critic": False},
                "physics_validation": {
                    "status": "passed",
                    "accepted_state_cache_manifest": str(case_dir / "artifacts/state_cache/manifest.json"),
                },
                "final_render": {
                    "required": True,
                    "status": "needed",
                    "attempts": 0,
                },
            }

        def current_timing(self):
            return SimpleNamespace(
                steps=2,
                render_fps=25,
                sim_dt=0.01,
                sim_substeps=1,
                render_every_n_steps=1,
                render_res=(64, 48),
                duration_sec=1.0,
                target_video_frames=2,
            )

        def load_json(self, path):
            return None

    return Session()
