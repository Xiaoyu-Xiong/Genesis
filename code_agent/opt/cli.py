"""Temporary command-line entry point for invoking the Opt agent.

This module should stay thin: parse CLI flags, build an `OptAgentRequest`, call
`run_opt_agent`, and print the structured result. Codex invocation, prompt
construction, report parsing, and report writing belong in `opt/agent.py`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from code_agent.configs import CONFIGS
from code_agent.opt.agent import run_opt_agent
from code_agent.opt.types import DEFAULT_PLANNER_INTENT, OptAgentRequest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Opt agent on one generated case.")
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--original-prompt", default=None)
    parser.add_argument("--planner-intent", default=DEFAULT_PLANNER_INTENT)
    parser.add_argument("--max-rollouts", type=int, default=None)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default=CONFIGS.opt.agent_backend)
    parser.add_argument("--timeout-sec", type=float, default=CONFIGS.opt.agent_timeout_sec)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--render-fps", type=int, default=None)
    parser.add_argument("--target-video-frames", type=int, default=None)
    parser.add_argument(
        "--render-baseline", action=argparse.BooleanOptionalAction, default=CONFIGS.opt.agent_render_baseline
    )
    parser.add_argument("--render-best", action=argparse.BooleanOptionalAction, default=CONFIGS.opt.agent_render_best)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    request = OptAgentRequest(
        case_dir=args.case_dir,
        original_prompt=args.original_prompt,
        planner_intent=args.planner_intent,
        max_rollouts=args.max_rollouts,
        backend=args.backend,
        timeout_sec=args.timeout_sec,
        render_baseline=args.render_baseline,
        render_best=args.render_best,
        steps=args.steps,
        duration_sec=args.duration_sec,
        render_fps=args.render_fps,
        target_video_frames=args.target_video_frames,
    )
    result = run_opt_agent(request)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
