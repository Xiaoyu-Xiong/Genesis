from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from code_agent.dataset import review_tui
from code_agent.dataset import agents
from code_agent.dataset.builder import _materialize_segments, build_dataset
from code_agent.dataset.media import (
    average_video_hash,
    build_contact_sheet,
    cut_clip,
    detect_scene_segments,
    probe_video,
    resolve_ytdlp_command,
    visual_signature,
)
from code_agent.dataset.models import BuildConfig, SegmentCandidate, SourceCandidate
from code_agent.dataset.review_tui import (
    case_line,
    resolve_review_start_index,
    reviewable_clip_positions,
)
from code_agent.dataset.seeds import collect_similarity_seeds
from code_agent.dataset.store import DatasetStore
from code_agent.dataset.utils import is_ytdlp_supported_url
from code_agent.utils.codex import CodexExecResult


def test_dataset_store_round_trip_reject_edit_and_export(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    store.add_source_video(
        manifest,
        {
            "id": "source_demo",
            "url": "https://example.test/demo.mp4",
            "status": "ready",
            "sha256": "abc",
            "path": "videos/source_demo.mp4",
        },
    )
    store.add_clip(
        manifest,
        {
            "id": "clip_demo",
            "source_video_id": "source_demo",
            "status": "accepted",
            "case_id": "old_case",
            "prompt": "Create a FEM+IPC scene inspired by an old demo. Render 10s behavior.",
            "clip_path": "clips/clip_demo.mp4",
            "clip_sha256": "cliphash",
            "visual_fingerprint": "ff00ff00ff00ff00",
            "frame_fingerprints": ["ff00ff00ff00ff00"],
            "color_histogram": [0.1] * 24,
            "foreground_component_fingerprints": ["ff00ff00ff00ff00"],
            "foreground_component_phashes": ["00ff00ff00ff00ff"],
            "foreground_color_histogram": [0.2] * 24,
            "prompt_revisions": [],
        },
    )
    store.save(manifest)

    loaded = store.load()
    assert loaded["source_videos"][0]["id"] == "source_demo"
    assert loaded["clips"][0]["clip_uri"].startswith("file://")
    assert loaded["clips"][0]["clip_uri"].endswith("/clips/clip_demo.mp4")
    assert list(loaded["clips"][0])[-5:] == [
        "frame_fingerprints",
        "color_histogram",
        "foreground_component_fingerprints",
        "foreground_component_phashes",
        "foreground_color_histogram",
    ]
    assert store.accepted_count(loaded) == 1
    assert store.clip_hash_seen(loaded, "cliphash")
    assert store.near_duplicate_clip(loaded, "ff00ff00ff00ff01")["id"] == "clip_demo"

    reject_event = store.reject_clip("clip_demo", reason="not suitable")
    assert reject_event["type"] == "reject"
    assert store.load()["clips"][0]["status"] == "rejected"

    edit_event = store.edit_clip(
        "clip_demo",
        prompt=(
            "soft_braid_demo|Create a FEM+IPC scene inspired by a soft braid demo: two rods twist, contact, "
            "and settle through real endpoint actuation. Render 10s behavior."
        ),
        reason="human rewrite",
    )
    edited = store.load()
    assert edit_event["type"] == "edit"
    assert edited["clips"][0]["case_id"] == "soft_braid_demo"
    assert edited["clips"][0]["status"] == "accepted"
    assert edited["style_memory"][0]["case_id"] == "soft_braid_demo"

    out_path = tmp_path / "cases.txt"
    count = store.export_cases(out_path)
    assert count == 1
    assert out_path.read_text(encoding="utf-8").startswith("soft_braid_demo|Create a FEM+IPC scene")


def test_dataset_store_detects_visual_near_duplicates_from_frame_signature(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    store.add_clip(
        manifest,
        {
            "id": "clip_a",
            "status": "accepted",
            "case_id": "clip_a",
            "prompt": "Create a rigid scene. Render 10s behavior.",
            "visual_fingerprint": "0000000000000000",
            "frame_fingerprints": [
                "ffffffff00000000",
                "fffffff000000000",
                "0fffffff00000000",
                "00ffffff00000000",
            ],
            "color_histogram": [0.05] * 24,
        },
    )

    duplicate = store.near_duplicate_clip(
        manifest,
        signature={
            "visual_fingerprint": "aaaaaaaaaaaaaaaa",
            "frame_fingerprints": [
                "ffffffff00000001",
                "fffffff000000001",
                "0fffffff00000001",
                "00ffffff00000001",
            ],
            "color_histogram": [0.05] * 24,
        },
    )

    assert duplicate["id"] == "clip_a"


def test_dataset_store_detects_visual_near_duplicates_from_foreground_components(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    store.add_clip(
        manifest,
        {
            "id": "clip_a",
            "status": "accepted",
            "case_id": "clip_a",
            "prompt": "Create a rigid scene. Render 10s behavior.",
            "visual_fingerprint": "0000000000000000",
            "frame_fingerprints": ["1111111111111111"],
            "color_histogram": [0.01] * 24,
            "foreground_component_fingerprints": [
                "f9f9f840086d4f4f",
                "3f0fcfe7f3f9fcfe",
            ],
            "foreground_component_phashes": [
                "49ceb5931fa4d291",
                "383898cfe7c2e9c4",
            ],
            "foreground_color_histogram": [0.05] * 24,
        },
    )

    candidates = store.duplicate_candidates(
        manifest,
        signature={
            "visual_fingerprint": "aaaaaaaaaaaaaaaa",
            "frame_fingerprints": ["2222222222222222"],
            "color_histogram": [0.02] * 24,
            "foreground_component_fingerprints": [
                "f9f9f840086d4f4e",
                "3f0fcfe7f3f9fcff",
            ],
            "foreground_component_phashes": [
                "49ceb5931fa4d290",
                "383898cfe7c2e9c5",
            ],
            "foreground_color_histogram": [0.05] * 24,
        },
    )

    assert candidates[0]["clip_id"] == "clip_a"
    assert candidates[0]["reason"] == "foreground_components"


def test_delete_duplicate_removes_clip_without_negative_memory(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    store.add_clip(
        manifest,
        {
            "id": "clip_dup",
            "status": "accepted",
            "case_id": "clip_dup",
            "prompt": "Create a rigid scene. Render 10s behavior.",
            "clip_path": "clips/clip_dup.mp4",
        },
    )
    store.save(manifest)

    event = store.delete_duplicate_clip("clip_dup", duplicate_of_clip_id="clip_original", reason="same demo")
    manifest = store.load()

    assert event["type"] == "delete_duplicate"
    assert event["negative_memory"] is False
    assert manifest["clips"][0]["status"] == "duplicate_deleted"
    assert manifest["clips"][0]["duplicate_of_clip_id"] == "clip_original"
    assert not manifest["style_memory"]
    assert all(item.get("type") != "reject" for item in manifest["review_events"])
    out_path = tmp_path / "cases.txt"
    assert store.export_cases(out_path) == 0


def test_delete_multi_example_removes_clip_without_negative_memory(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    store.add_clip(
        manifest,
        {
            "id": "clip_mixed",
            "status": "accepted",
            "case_id": "clip_mixed",
            "prompt": "Create a rigid scene. Render 10s behavior.",
            "clip_path": "clips/clip_mixed.mp4",
        },
    )
    store.save(manifest)

    event = store.delete_multi_example_clip("clip_mixed", reason="contains two demos")
    manifest = store.load()

    assert event["type"] == "delete_multi_example"
    assert event["negative_memory"] is False
    assert manifest["clips"][0]["status"] == "multi_example_deleted"
    assert manifest["clips"][0]["multi_example"] is True
    assert all(item.get("type") != "reject" for item in manifest["review_events"])
    assert store.export_cases(tmp_path / "cases.txt") == 0


def test_set_clip_category_records_review_event(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    store.add_clip(
        manifest,
        {
            "id": "clip_category",
            "status": "accepted",
            "case_id": "clip_category",
            "prompt": "Create a FEM+IPC scene inspired by a soft body demo. Render 10s behavior.",
            "clip_path": "clips/clip_category.mp4",
        },
    )
    store.save(manifest)

    event = store.set_clip_category("clip_category", category="deformable bodies", reason="review label")
    manifest = store.load()

    assert event["type"] == "set_category"
    assert event["before_category"] is None
    assert event["after_category"] == "deformable_bodies"
    assert manifest["clips"][0]["category"] == "deformable_bodies"
    assert manifest["clips"][0]["status"] == "accepted"
    assert manifest["review_events"][-1]["reason"] == "review label"


def test_review_tui_start_position_helpers_follow_last_reviewed_clip(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    for clip_id in ("clip_a", "clip_b", "clip_c"):
        store.add_clip(
            manifest,
            {
                "id": clip_id,
                "status": "accepted",
                "case_id": clip_id,
                "prompt": f"Create a FEM+IPC scene inspired by {clip_id}. Render 10s behavior.",
            },
        )

    assert [clip["id"] for _index, clip in reviewable_clip_positions(manifest)] == ["clip_a", "clip_b", "clip_c"]
    assert resolve_review_start_index(manifest) == 0
    assert resolve_review_start_index(manifest, "2") == 1
    assert resolve_review_start_index(manifest, "clip_c") == 2
    assert case_line(manifest["clips"][0]).startswith("clip_a|Create a FEM+IPC scene")

    manifest["review_state"] = {
        "last_reviewed_clip_id": "clip_a",
        "last_reviewed_manifest_index": 0,
    }
    assert resolve_review_start_index(manifest) == 1

    manifest["review_state"] = {
        "last_reviewed_clip_id": "clip_c",
        "last_reviewed_manifest_index": 2,
    }
    assert resolve_review_start_index(manifest) == 3


def test_review_actions_accept_record_position_and_delete_truncated(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    store.add_clip(
        manifest,
        {
            "id": "clip_review",
            "status": "accepted",
            "case_id": "clip_review",
            "prompt": "Create a FEM+IPC scene inspired by a review demo. Render 10s behavior.",
            "clip_path": "clips/clip_review.mp4",
        },
    )
    store.save(manifest)

    accept_event = store.accept_clip("clip_review", reason="looks good")
    state = store.record_review_position("clip_review", manifest_index=0, note="accept")
    manifest = store.load()

    assert accept_event["type"] == "accept"
    assert manifest["clips"][0]["status"] == "accepted"
    assert manifest["clips"][0]["reviewed_at"]
    assert manifest["clips"][0]["accept_reason"] == "looks good"
    assert state["last_reviewed_clip_id"] == "clip_review"
    assert manifest["review_state"]["last_reviewed_manifest_index"] == 0

    truncated_event = store.delete_truncated_clip("clip_review", reason="cut from the middle")
    manifest = store.load()

    assert truncated_event["type"] == "delete_truncated"
    assert truncated_event["negative_memory"] is False
    assert manifest["clips"][0]["status"] == "truncated_deleted"
    assert manifest["clips"][0]["truncated"] is True
    assert all(item.get("type") != "reject" for item in manifest["review_events"])


def test_review_tui_prefers_windows_file_opener_on_wsl(tmp_path, monkeypatch):
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"demo")

    def fake_which(name):
        return {
            "xdg-open": "/usr/bin/xdg-open",
            "explorer.exe": "/mnt/c/Windows/explorer.exe",
        }.get(name)

    monkeypatch.setattr(review_tui.shutil, "which", fake_which)
    monkeypatch.setattr(review_tui, "_wsl_windows_path", lambda path: r"\\wsl.localhost\Ubuntu\tmp\clip.mp4")

    label, command = review_tui._open_command_for_path(clip_path, env={"WSL_DISTRO_NAME": "Ubuntu"})

    assert label == "explorer.exe"
    assert command == ["/mnt/c/Windows/explorer.exe", r"\\wsl.localhost\Ubuntu\tmp\clip.mp4"]


def test_review_tui_defaults_to_modern_waiting_editor(monkeypatch):
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)

    def fake_which(name):
        return {"code": "/usr/bin/code", "vi": "/usr/bin/vi"}.get(name)

    monkeypatch.setattr(review_tui.shutil, "which", fake_which)

    command = review_tui._resolve_editor_command(None, env={})

    assert command == "/usr/bin/code --wait --reuse-window"


def test_bilibili_urls_are_routed_to_ytdlp():
    assert is_ytdlp_supported_url("https://www.bilibili.com/video/BV1964y1275C/")
    assert is_ytdlp_supported_url("https://b23.tv/example")


def test_failed_source_urls_are_retryable(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    url = "https://www.youtube.com/watch?v=demo"
    store.add_source_video(manifest, {"id": "failed_demo", "url": url, "status": "failed"})
    assert not store.source_url_seen(manifest, url)

    store.add_source_video(manifest, {"id": "ready_demo", "url": url, "status": "ready"})
    assert store.source_url_seen(manifest, url)


def test_ytdlp_resolver_prefers_current_python_module():
    assert resolve_ytdlp_command()[1:] == ["-m", "yt_dlp"]


def test_curator_memory_marks_failed_sources_retryable(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    retry_url = "https://www.youtube.com/watch?v=retry"
    ready_url = "https://www.youtube.com/watch?v=ready"
    manifest["source_videos"].extend(
        [
            {"id": "failed_source", "url": retry_url, "status": "failed", "error": "old yt-dlp failed"},
            {"id": "ready_source", "url": ready_url, "status": "ready", "sha256": "abc"},
        ]
    )

    prompt = agents._curator_prompt(  # noqa: SLF001 - this is the unit boundary for prompt policy.
        candidates=[
            SourceCandidate(candidate_id="retry", video_url=retry_url, title="Retry"),
            SourceCandidate(candidate_id="ready", video_url=ready_url, title="Ready"),
        ],
        manifest=manifest,
        similarity_seeds=[],
    )

    assert "retryable" in prompt
    assert "retryable_failed_sources" in prompt
    assert "old yt-dlp failed" in prompt


def test_codex_agent_wrappers_parse_mocked_schema_outputs(tmp_path, monkeypatch):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()

    def fake_run_codex_exec(request):
        payload = _payload_for_role(request.role)
        request.final_message_path.parent.mkdir(parents=True, exist_ok=True)
        request.final_message_path.write_text(json.dumps(payload), encoding="utf-8")
        request.output_jsonl_path.write_text("", encoding="utf-8")
        stderr_path = request.output_jsonl_path.with_suffix(request.output_jsonl_path.suffix + ".stderr")
        stderr_path.write_text("", encoding="utf-8")
        return CodexExecResult(
            role=request.role,
            success=True,
            exit_code=0,
            duration_sec=0.01,
            command=["codex"],
            cwd=str(request.cwd),
            sandbox=request.sandbox,
            output_jsonl_path=str(request.output_jsonl_path),
            final_message_path=str(request.final_message_path),
            output_schema_path=str(request.output_schema_path),
            codex_version="codex-test",
            stderr_path=str(stderr_path),
            started_at_unix=time.time(),
            ended_at_unix=time.time(),
        )

    monkeypatch.setattr(agents, "run_codex_exec", fake_run_codex_exec)
    logs_dir = tmp_path / "logs"

    candidates = agents.scout_sources(
        store=store,
        manifest=manifest,
        sources=[],
        needed_clips=1,
        logs_dir=logs_dir,
    )
    assert candidates[0].candidate_id == "cand_1"

    curated = agents.curate_sources(candidates=candidates, manifest=manifest, logs_dir=logs_dir)
    assert curated == candidates

    segments = agents.segment_video(
        source_record={"id": "source_demo", "duration_sec": 2.0},
        deterministic_segments=[],
        timeline_sheet=tmp_path / "missing.jpg",
        logs_dir=logs_dir,
    )
    assert segments[0].title_slug == "soft_demo"

    case_id, prompt = agents.write_prompt(
        clip_record={"id": "clip_demo", "start_sec": 0.0, "end_sec": 2.0, "visual_summary": "soft demo"},
        manifest=manifest,
        clip_sheet=tmp_path / "missing.jpg",
        logs_dir=logs_dir,
    )
    assert case_id == "soft_demo"
    assert prompt.startswith("Create a FEM+IPC scene inspired by")

    duplicate_review = agents.review_duplicate_clip(
        clip_record={"id": "clip_demo", "title": "soft demo"},
        duplicate_candidate={
            "clip": {"id": "clip_existing", "case_id": "soft_demo"},
            "score": 0.84,
            "reason": "foreground_components",
            "metrics": {},
        },
        current_sheet=tmp_path / "missing_current.jpg",
        existing_sheet=tmp_path / "missing_existing.jpg",
        logs_dir=logs_dir,
    )
    assert duplicate_review["decision"] == "duplicate"
    assert duplicate_review["duplicate_of_clip_id"] == "clip_existing"


def test_similarity_seeds_collect_from_cli_file_and_manifest(tmp_path):
    cases_path = tmp_path / "cases.txt"
    cases_path.write_text(
        "\n".join(
            [
                "soft_twist|Create a FEM+IPC scene inspired by a twisting rods demo. Render 10s behavior.",
                "soft_squeeze|Create a FEM+IPC scene inspired by a squeeze-through-gap demo. Render 10s behavior.",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "clips": [
            {
                "id": "clip_1",
                "status": "accepted",
                "case_id": "accepted_case",
                "prompt": "Create a FEM+IPC scene inspired by a refined accepted clip. Render 10s behavior.",
            }
        ],
        "style_memory": [
            {
                "clip_id": "clip_2",
                "case_id": "edited_case",
                "prompt": "Create a FEM+IPC scene inspired by a human edited prompt. Render 10s behavior.",
                "reason": "better wording",
            }
        ],
    }

    seeds = collect_similarity_seeds(
        BuildConfig(
            target_clips=1,
            data_root=tmp_path / "dataset",
            similar_to=("soft_cli|Create a FEM+IPC scene inspired by a CLI seed. Render 10s behavior.",),
            similar_to_file=cases_path,
            similarity_seed_limit=10,
        ),
        manifest,
    )

    prompts = [seed.prompt for seed in seeds]
    assert any("CLI seed" in prompt for prompt in prompts)
    assert any("twisting rods" in prompt for prompt in prompts)
    assert any("human edited prompt" in prompt for prompt in prompts)
    assert any(seed.case_id == "accepted_case" for seed in seeds)


def test_scout_prompt_receives_similarity_seeds(tmp_path, monkeypatch):
    store = DatasetStore(tmp_path / "dataset")
    manifest = store.empty_manifest()
    captured_prompt = {}

    def fake_run_codex_exec(request):
        captured_prompt["text"] = request.prompt
        payload = _payload_for_role(request.role)
        request.final_message_path.parent.mkdir(parents=True, exist_ok=True)
        request.final_message_path.write_text(json.dumps(payload), encoding="utf-8")
        request.output_jsonl_path.write_text("", encoding="utf-8")
        stderr_path = request.output_jsonl_path.with_suffix(request.output_jsonl_path.suffix + ".stderr")
        stderr_path.write_text("", encoding="utf-8")
        return CodexExecResult(
            role=request.role,
            success=True,
            exit_code=0,
            duration_sec=0.01,
            command=["codex"],
            cwd=str(request.cwd),
            sandbox=request.sandbox,
            output_jsonl_path=str(request.output_jsonl_path),
            final_message_path=str(request.final_message_path),
            output_schema_path=str(request.output_schema_path),
            codex_version="codex-test",
            stderr_path=str(stderr_path),
            started_at_unix=time.time(),
            ended_at_unix=time.time(),
        )

    monkeypatch.setattr(agents, "run_codex_exec", fake_run_codex_exec)
    seeds = collect_similarity_seeds(
        BuildConfig(
            target_clips=1,
            data_root=store.root,
            similar_to=(
                "soft_braid|Create a FEM+IPC scene inspired by a braided rod contact demo. Render 10s behavior.",
            ),
        ),
        manifest,
    )

    agents.scout_sources(
        store=store,
        manifest=manifest,
        sources=[],
        needed_clips=1,
        logs_dir=tmp_path / "logs",
        similarity_seeds=seeds,
    )

    assert "Similarity targets from previously tuned successful prompts" in captured_prompt["text"]
    assert "braided rod contact demo" in captured_prompt["text"]


def test_segment_and_prompt_writer_prompts_warn_against_truncated_examples(tmp_path, monkeypatch):
    captured_prompts: list[str] = []

    def fake_run_codex_exec(request):
        captured_prompts.append(request.prompt)
        payload = _payload_for_role(request.role)
        request.final_message_path.parent.mkdir(parents=True, exist_ok=True)
        request.final_message_path.write_text(json.dumps(payload), encoding="utf-8")
        request.output_jsonl_path.write_text("", encoding="utf-8")
        stderr_path = request.output_jsonl_path.with_suffix(request.output_jsonl_path.suffix + ".stderr")
        stderr_path.write_text("", encoding="utf-8")
        return CodexExecResult(
            role=request.role,
            success=True,
            exit_code=0,
            duration_sec=0.01,
            command=["codex"],
            cwd=str(request.cwd),
            sandbox=request.sandbox,
            output_jsonl_path=str(request.output_jsonl_path),
            final_message_path=str(request.final_message_path),
            output_schema_path=str(request.output_schema_path),
            codex_version="codex-test",
            stderr_path=str(stderr_path),
            started_at_unix=time.time(),
            ended_at_unix=time.time(),
        )

    monkeypatch.setattr(agents, "run_codex_exec", fake_run_codex_exec)
    logs_dir = tmp_path / "logs"

    agents.segment_video(
        source_record={"id": "source_demo", "duration_sec": 4.0},
        deterministic_segments=[],
        timeline_sheet=tmp_path / "missing_timeline.jpg",
        logs_dir=logs_dir,
    )
    agents.write_prompt(
        clip_record={"id": "clip_demo", "start_sec": 1.0, "end_sec": 3.0, "visual_summary": "partial demo"},
        manifest={},
        clip_sheet=tmp_path / "missing_clip.jpg",
        logs_dir=logs_dir,
    )

    segment_prompt, writer_prompt = captured_prompts
    assert "Do not cut a complete demo from the middle" in segment_prompt
    assert "omit that segment rather than" in segment_prompt
    assert "returning a partial mid-demo clip" in segment_prompt
    assert "starts after the main setup" in writer_prompt
    assert "cuts off before" in writer_prompt
    assert "the main outcome/settling" in writer_prompt


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for video tests")
def test_media_helpers_segment_and_clip_synthetic_video(tmp_path):
    video_path = _make_two_color_video(tmp_path / "two_color.mp4")

    info = probe_video(video_path)
    assert info.duration_sec >= 1.5

    sheet_path = tmp_path / "sheet.jpg"
    build_contact_sheet(video_path, sheet_path, max_frames=4, thumb_width=80)
    assert sheet_path.exists()

    segments = detect_scene_segments(
        video_path, source_id="two_color", threshold=0.2, min_segment_sec=0.5, sample_fps=4
    )
    assert len(segments) >= 2

    clip_path = tmp_path / "clip.mp4"
    cut_clip(video_path, start_sec=segments[0].start_sec, end_sec=segments[0].end_sec, out_path=clip_path)
    assert clip_path.exists()
    assert average_video_hash(clip_path)
    signature = visual_signature(clip_path)
    assert signature["visual_fingerprint"]
    assert signature["frame_fingerprints"]
    assert signature["signature_version"] == 2
    assert "foreground_component_fingerprints" in signature


def test_real_rigid_ipc_duplicate_pairs_detected_by_cv_when_available(tmp_path):
    pairs = [
        (
            Path(
                "code_agent/dataset/data/clips/"
                "intersection_free_rigid_body_dynamics_main_video_anchor_chain_threading_disc.mp4"
            ),
            Path(
                "code_agent/dataset/data/clips/"
                "intersection_free_rigid_body_dynamics_supplemental_video_intersection_free_rigid_75343efa11.mp4"
            ),
        ),
        (
            Path(
                "code_agent/dataset/data/clips/"
                "intersection_free_rigid_body_dynamics_main_video_sliding_plate_ramp_comparison.mp4"
            ),
            Path(
                "code_agent/dataset/data/clips/"
                "intersection_free_rigid_body_dynamics_supplemental_video_intersection_free_rigid_3dfe7b75c2.mp4"
            ),
        ),
    ]
    missing = [path for pair in pairs for path in pair if not path.exists()]
    if missing:
        pytest.skip("real dataset calibration clips are not present")

    store = DatasetStore(tmp_path / "dataset")
    for index, (existing_path, current_path) in enumerate(pairs, start=1):
        manifest = store.empty_manifest()
        existing_signature = visual_signature(existing_path)
        store.add_clip(
            manifest,
            {
                "id": f"existing_{index}",
                "status": "accepted",
                "case_id": f"existing_{index}",
                "prompt": "Create a rigid contact scene. Render 10s behavior.",
                **existing_signature,
            },
        )
        candidates = store.duplicate_candidates(manifest, signature=visual_signature(current_path))
        assert candidates, f"{existing_path.name} vs {current_path.name} should be a CV duplicate candidate"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for video tests")
def test_materialize_segments_deletes_codex_rejected_duplicate_artifacts(tmp_path, monkeypatch):
    video_path = _make_two_color_video(tmp_path / "source.mp4")
    store = DatasetStore(tmp_path / "dataset")
    store.ensure_dirs()
    manifest = store.empty_manifest()
    source_record = {
        "id": "source_demo",
        "url": str(video_path),
        "duration_sec": 2.0,
    }

    def fake_duplicate_candidates(
        self, manifest, fingerprint=None, *, signature=None, max_results=3, exclude_clip_id=None
    ):
        return [
            {
                "clip": {"id": "existing_clip", "contact_sheet_path": None},
                "clip_id": "existing_clip",
                "score": 0.9,
                "reason": "test_duplicate",
                "metrics": {},
            }
        ]

    def fake_review_duplicate_clip(**kwargs):
        return {
            "decision": "duplicate",
            "duplicate_of_clip_id": "existing_clip",
            "reason": "same demo",
            "confidence": 0.99,
        }

    def fail_write_prompt(**kwargs):
        raise AssertionError("duplicate clips should be skipped before prompt writing")

    monkeypatch.setattr(DatasetStore, "duplicate_candidates", fake_duplicate_candidates)
    monkeypatch.setattr(agents, "review_duplicate_clip", fake_review_duplicate_clip)
    monkeypatch.setattr(agents, "write_prompt", fail_write_prompt)

    added = _materialize_segments(
        store=store,
        manifest=manifest,
        source_record=source_record,
        source_path=video_path,
        segments=[
            SegmentCandidate(
                title_slug="duplicate_segment",
                start_sec=0.0,
                end_sec=1.0,
                visual_summary="duplicate segment",
                reason="test",
            )
        ],
        target=1,
        logs_dir=tmp_path / "logs",
        similarity_seeds=[],
        run_codex=True,
    )

    assert added == 0
    assert not list(store.clips_dir.glob("*.mp4"))
    assert not list(store.contact_sheets_dir.glob("*.jpg"))
    assert not manifest["clips"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for video tests")
def test_build_dataset_no_codex_from_explicit_local_video_is_incremental(tmp_path):
    video_path = _make_two_color_video(tmp_path / "explicit.mp4")
    data_root = tmp_path / "dataset"

    summary = build_dataset(
        BuildConfig(
            target_clips=1,
            data_root=data_root,
            sources=(str(video_path),),
            run_codex=False,
        )
    )
    assert summary.status == "complete"
    assert summary.accepted_clips == 1
    assert summary.clips_added == 1

    second_summary = build_dataset(
        BuildConfig(
            target_clips=1,
            data_root=data_root,
            sources=(str(video_path),),
            run_codex=False,
        )
    )
    assert second_summary.status == "already_complete"
    assert second_summary.clips_added == 0

    manifest = DatasetStore(data_root).load()
    accepted = [clip for clip in manifest["clips"] if clip["status"] == "accepted"]
    assert accepted[0]["prompt"].startswith("Create a FEM+IPC scene inspired by")
    assert accepted[0]["clip_uri"].startswith("file://")
    assert accepted[0]["clip_uri"].endswith(".mp4")


def test_build_dataset_direct_sources_skip_scout_but_keep_codex_steps(tmp_path, monkeypatch):
    video_path = _make_two_color_video(tmp_path / "explicit.mp4")
    data_root = tmp_path / "dataset"
    called: list[str] = []

    def fail_scout(**kwargs):
        raise AssertionError("Direct source builds should not call scout")

    def fake_curate_sources(**kwargs):
        called.append("curator")
        return kwargs["candidates"]

    def fake_segment_video(**kwargs):
        called.append("segmenter")
        return [
            SegmentCandidate(
                title_slug="single_demo",
                start_sec=0.0,
                end_sec=2.0,
                visual_summary="one complete synthetic demo",
                reason="test segment",
                confidence=1.0,
            )
        ]

    def fake_write_prompt(**kwargs):
        called.append("prompt_writer")
        return ("single_demo", "Create a FEM+IPC scene inspired by a synthetic contact demo. Render 2s behavior.")

    monkeypatch.setattr(agents, "scout_sources", fail_scout)
    monkeypatch.setattr(agents, "curate_sources", fake_curate_sources)
    monkeypatch.setattr(agents, "segment_video", fake_segment_video)
    monkeypatch.setattr(agents, "write_prompt", fake_write_prompt)

    summary = build_dataset(
        BuildConfig(
            target_clips=1,
            data_root=data_root,
            sources=(str(video_path),),
            run_codex=True,
        )
    )

    assert summary.status == "complete"
    assert called == ["curator", "segmenter", "prompt_writer"]


def _payload_for_role(role: str) -> dict[str, object]:
    if role == "dataset_scout":
        return {
            "candidate_sources": [
                {
                    "candidate_id": "cand_1",
                    "video_url": "https://example.test/demo.mp4",
                    "title": "Demo",
                    "project_url": None,
                    "paper_title": "Paper",
                    "venue": "SIGGRAPH",
                    "source_url": "https://example.test/project",
                    "license_notes": None,
                    "source_policy_notes": "public project page",
                    "notes": "good deformable demo",
                    "confidence": 0.9,
                }
            ]
        }
    if role == "dataset_curator":
        return {
            "decisions": [
                {"candidate_id": "cand_1", "status": "accept", "reason": "usable", "avoid_similarity_note": None}
            ]
        }
    if role.startswith("dataset_segmenter"):
        return {
            "segments": [
                {
                    "title_slug": "soft_demo",
                    "start_sec": 0.0,
                    "end_sec": 2.0,
                    "visual_summary": "soft body squeezes through a gap",
                    "reason": "one self-contained demo",
                    "confidence": 0.8,
                }
            ]
        }
    if role.startswith("dataset_duplicate_reviewer"):
        return {
            "decision": "duplicate",
            "duplicate_of_clip_id": "clip_existing",
            "reason": "same dominant visual demo",
            "confidence": 0.9,
        }
    return {
        "case_id": "soft_demo",
        "prompt": (
            "Create a FEM+IPC scene inspired by a soft body squeeze demo: a hyperelastic toy is pushed through a "
            "narrow rigid gap, visibly squashes, slides, rebounds, and settles through contact. Render 10s behavior."
        ),
        "coverage": "deformable",
        "notes": None,
    }


def _make_two_color_video(path: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=160x120:d=1:r=24",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x120:d=1:r=24",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p[v]",
            "-map",
            "[v]",
            str(path),
        ],
        check=True,
        timeout=30,
    )
    return path
