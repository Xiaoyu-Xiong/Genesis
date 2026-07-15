from __future__ import annotations

import argparse
import json
from pathlib import Path

from code_agent.dataset.builder import build_dataset
from code_agent.dataset.models import BuildConfig
from code_agent.dataset.review_tui import run_review_tui
from code_agent.dataset.store import DEFAULT_DATA_ROOT, DatasetStore
from code_agent.io_utils import load_json_object
from code_agent.utils.codex import CodexAuthFreshnessError, ensure_configured_codex_accounts_fresh


def _cmd_build(args: argparse.Namespace) -> None:
    if not args.no_codex:
        _ensure_codex_auth_fresh()
    summary = build_dataset(
        BuildConfig(
            target_clips=args.target_clips,
            data_root=args.data_root,
            sources=tuple(args.source or ()),
            source_file=args.source_file,
            similar_to=tuple(args.similar_to or ()),
            similar_to_file=args.similar_to_file,
            similarity_seed_limit=args.similarity_seed_limit,
            max_candidates=args.max_candidates,
            max_downloads=args.max_downloads,
            max_scout_rounds=args.max_scout_rounds,
            max_empty_scout_rounds=args.max_empty_scout_rounds,
            run_codex=not args.no_codex,
        )
    )
    print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))


def _cmd_status(args: argparse.Namespace) -> None:
    store = DatasetStore(args.data_root)
    print(json.dumps(store.status_summary(), indent=2, ensure_ascii=False))


def _ensure_codex_auth_fresh() -> None:
    try:
        ensure_configured_codex_accounts_fresh()
    except CodexAuthFreshnessError as exc:
        raise SystemExit(str(exc)) from exc


def _cmd_reject(args: argparse.Namespace) -> None:
    store = DatasetStore(args.data_root)
    event = store.reject_clip(args.clip_id, reason=args.reason, avoid_similarity_note=args.avoid_similarity_note)
    print(json.dumps(event, indent=2, ensure_ascii=False))


def _cmd_edit(args: argparse.Namespace) -> None:
    store = DatasetStore(args.data_root)
    event = store.edit_clip(args.clip_id, prompt=args.prompt, reason=args.reason)
    print(json.dumps(event, indent=2, ensure_ascii=False))


def _cmd_set_category(args: argparse.Namespace) -> None:
    store = DatasetStore(args.data_root)
    event = store.set_clip_category(args.clip_id, category=args.category, reason=args.reason)
    print(json.dumps(event, indent=2, ensure_ascii=False))


def _cmd_set_split(args: argparse.Namespace) -> None:
    store = DatasetStore(args.data_root)
    event = store.set_clip_split(args.clip_id, split=args.split, reason=args.reason)
    print(json.dumps(event, indent=2, ensure_ascii=False))


def _cmd_assign_splits(args: argparse.Namespace) -> None:
    store = DatasetStore(args.data_root)
    summary = store.update_manifest(
        lambda manifest: store.assign_train_test_splits(
            manifest,
            test_fraction=args.test_fraction,
            temporary=True,
            include_unset=not args.no_include_unset,
            overwrite_permanent=False,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_assign_final_splits(args: argparse.Namespace) -> None:
    store = DatasetStore(args.data_root)
    summary = store.update_manifest(
        lambda manifest: store.assign_train_test_splits(
            manifest,
            test_fraction=args.test_fraction,
            temporary=False,
            include_unset=True,
            overwrite_permanent=True,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_finalize_tmp_splits(args: argparse.Namespace) -> None:
    store = DatasetStore(args.data_root)
    summary = store.update_manifest(store.finalize_tmp_splits)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_split_summary(args: argparse.Namespace) -> None:
    store = DatasetStore(args.data_root)
    print(json.dumps(store.split_summary(), indent=2, ensure_ascii=False))


def _cmd_drop_paper_prompts(args: argparse.Namespace) -> None:
    store = DatasetStore(args.data_root)
    summary = store.update_manifest(store.drop_paper_prompts)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_make_run_batch(args: argparse.Namespace) -> None:
    store = DatasetStore(args.data_root)
    summary = store.make_run_batch(
        mode=args.mode,
        count=args.count,
        out_path=args.out,
        seed=args.seed,
        mark_trained=not args.no_mark_trained,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_mark_train_results(args: argparse.Namespace) -> None:
    store = DatasetStore(args.data_root)
    summary = load_json_object(args.summary)
    if not isinstance(summary, dict):
        raise SystemExit(f"Unable to load suite summary JSON object: {args.summary}")
    result = store.mark_train_results_from_suite(summary, summary_path=args.summary)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _cmd_delete_duplicate(args: argparse.Namespace) -> None:
    store = DatasetStore(args.data_root)
    event = store.delete_duplicate_clip(
        args.clip_id,
        duplicate_of_clip_id=args.duplicate_of,
        reason=args.reason,
    )
    print(json.dumps(event, indent=2, ensure_ascii=False))


def _cmd_delete_multi_example(args: argparse.Namespace) -> None:
    store = DatasetStore(args.data_root)
    event = store.delete_multi_example_clip(args.clip_id, reason=args.reason)
    print(json.dumps(event, indent=2, ensure_ascii=False))


def _cmd_review_tui(args: argparse.Namespace) -> None:
    run_review_tui(
        args.data_root,
        start=args.start,
        editor=args.editor,
        auto_play=not args.no_auto_play,
    )


def _cmd_export_cases(args: argparse.Namespace) -> None:
    store = DatasetStore(args.data_root)
    count = store.export_cases(args.out)
    print(f"Exported {count} accepted cases to {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m code_agent.dataset.cli")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Incrementally build video-clip to code_agent-prompt pairs.")
    build.add_argument("--target-clips", type=int, required=True)
    build.add_argument(
        "--source", action="append", help="Explicit project page, direct video URL, or local video path."
    )
    build.add_argument("--source-file", type=Path, default=None)
    build.add_argument(
        "--similar-to",
        action="append",
        help="Previously tuned case prompt, case_id|prompt line, or behavior description to search around.",
    )
    build.add_argument(
        "--similar-to-file",
        type=Path,
        default=None,
        help="Text cases file or JSON manifest/summary containing successful prompts to use as similarity targets.",
    )
    build.add_argument(
        "--similarity-seed-limit",
        type=int,
        default=12,
        help="Maximum number of similarity target prompts injected into Codex subagents.",
    )
    build.add_argument("--max-candidates", type=int, default=None)
    build.add_argument("--max-downloads", type=int, default=None)
    build.add_argument(
        "--max-scout-rounds",
        type=int,
        default=None,
        help=(
            "Maximum Codex scout rounds before stopping. By default this scales with remaining target size "
            "and is capped at 20."
        ),
    )
    build.add_argument(
        "--max-empty-scout-rounds",
        type=int,
        default=3,
        help="Stop after this many consecutive rounds with no fresh candidate URLs to try.",
    )
    build.add_argument("--no-codex", action="store_true", help="Use only explicit sources and deterministic fallbacks.")
    build.set_defaults(func=_cmd_build)

    status = sub.add_parser("status", help="Print dataset manifest status.")
    status.set_defaults(func=_cmd_status)

    reject = sub.add_parser("reject", help="Reject a clip and record future-avoidance memory.")
    reject.add_argument("--clip-id", required=True)
    reject.add_argument("--reason", required=True)
    reject.add_argument("--avoid-similarity-note", default=None)
    reject.set_defaults(func=_cmd_reject)

    edit = sub.add_parser("edit", help="Edit a clip prompt and add the edit to style memory.")
    edit.add_argument("--clip-id", required=True)
    edit.add_argument("--prompt", required=True)
    edit.add_argument("--reason", default=None)
    edit.set_defaults(func=_cmd_edit)

    category = sub.add_parser("set-category", help="Set a broad category label on a clip.")
    category.add_argument("--clip-id", required=True)
    category.add_argument(
        "--category",
        required=True,
        help="Broad clip category label: rigid, deformable bodies, or cloth.",
    )
    category.add_argument("--reason", default="human category label")
    category.set_defaults(func=_cmd_set_category)

    split = sub.add_parser("set-split", help="Set a clip's dataset split label.")
    split.add_argument("--clip-id", required=True)
    split.add_argument("--split", required=True, choices=("train", "test", "train-tmp", "test-tmp"))
    split.add_argument("--reason", default="human split label")
    split.set_defaults(func=_cmd_set_split)

    assign_splits = sub.add_parser(
        "assign-splits",
        help="Alias for assign-tmp-splits: assign train-tmp/test-tmp labels without touching permanent labels.",
    )
    assign_splits.add_argument("--test-fraction", type=float, default=0.30)
    assign_splits.add_argument("--no-include-unset", action="store_true")
    assign_splits.set_defaults(func=_cmd_assign_splits)

    assign_tmp_splits = sub.add_parser(
        "assign-tmp-splits",
        help="Reassign all temporary/unset accepted clips to train-tmp/test-tmp.",
    )
    assign_tmp_splits.add_argument("--test-fraction", type=float, default=0.30)
    assign_tmp_splits.add_argument("--no-include-unset", action="store_true")
    assign_tmp_splits.set_defaults(func=_cmd_assign_splits)

    finalize_tmp_splits = sub.add_parser(
        "finalize-tmp-splits",
        help="Promote train-tmp/test-tmp labels to permanent train/test labels.",
    )
    finalize_tmp_splits.set_defaults(func=_cmd_finalize_tmp_splits)

    assign_final_splits = sub.add_parser(
        "assign-final-splits",
        help="Reassign all accepted clips directly to permanent train/test labels.",
    )
    assign_final_splits.add_argument("--test-fraction", type=float, default=0.30)
    assign_final_splits.set_defaults(func=_cmd_assign_final_splits)

    split_summary = sub.add_parser("split-summary", help="Print train/test split summary.")
    split_summary.set_defaults(func=_cmd_split_summary)

    drop_paper_prompts = sub.add_parser(
        "drop-paper-prompts",
        help="Remove deprecated prompt_from_paper blocks from clip records.",
    )
    drop_paper_prompts.set_defaults(func=_cmd_drop_paper_prompts)

    run_batch = sub.add_parser(
        "make-run-batch",
        help="Export a train/test batch. Train batches advance through untrained permanent train data in order.",
    )
    run_batch.add_argument("--mode", required=True, choices=("train", "test"))
    run_batch.add_argument("--count", type=int, required=True)
    run_batch.add_argument("--out", type=Path, required=True)
    run_batch.add_argument("--seed", type=int, default=None, help="Random seed for test mode.")
    run_batch.add_argument(
        "--no-mark-trained",
        action="store_true",
        help="Deprecated compatibility flag; train clips are now marked only after pass results are synced.",
    )
    run_batch.set_defaults(func=_cmd_make_run_batch)

    mark_train_results = sub.add_parser(
        "mark-train-results",
        help="Mark train clips as trained only for cases that pass in a run-suite summary.",
    )
    mark_train_results.add_argument("--summary", type=Path, required=True)
    mark_train_results.set_defaults(func=_cmd_mark_train_results)

    delete_duplicate = sub.add_parser(
        "delete-duplicate",
        help="Remove a leaked duplicate from the active dataset without adding negative rejection memory.",
    )
    delete_duplicate.add_argument("--clip-id", required=True)
    delete_duplicate.add_argument("--duplicate-of", default=None)
    delete_duplicate.add_argument("--reason", default=None)
    delete_duplicate.set_defaults(func=_cmd_delete_duplicate)

    delete_multi_example = sub.add_parser(
        "delete-multi-example",
        help="Remove a clip that contains multiple independent examples without adding negative rejection memory.",
    )
    delete_multi_example.add_argument("--clip-id", required=True)
    delete_multi_example.add_argument("--reason", default=None)
    delete_multi_example.set_defaults(func=_cmd_delete_multi_example)

    review_tui = sub.add_parser("review-tui", help="Interactively review accepted dataset clips.")
    review_tui.add_argument(
        "--start",
        default=None,
        help="1-based accepted item index or accepted clip id. Defaults to after the last reviewed clip.",
    )
    review_tui.add_argument(
        "--editor",
        default=None,
        help="Editor command for prompt edits; defaults to $VISUAL/$EDITOR, then code/cursor/nano/vi.",
    )
    review_tui.add_argument(
        "--no-auto-play",
        action="store_true",
        help="Do not automatically open the clip video when moving to a new example.",
    )
    review_tui.set_defaults(func=_cmd_review_tui)

    export = sub.add_parser("export-cases", help="Export accepted clips as case_id|prompt lines.")
    export.add_argument("--out", type=Path, required=True)
    export.set_defaults(func=_cmd_export_cases)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
