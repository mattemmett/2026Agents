from __future__ import annotations

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Agentic devcontainer healthcheck")

    p.add_argument(
        "--only",
        default=None,
        help="Comma-separated checks to run (weaviate,postgres,redis)",
    )
    p.add_argument(
        "--skip",
        default=None,
        help="Comma-separated checks to skip (weaviate,postgres,redis)",
    )
    p.add_argument(
        "--llm-plan",
        action="store_true",
        help="Use a constrained LLM to propose the todo order/subset",
    )
    p.add_argument(
        "--debug-plan",
        action="store_true",
        help="Print raw/parsed LLM planner output in the report",
    )

    # --- Human-in-the-loop ---
    p.add_argument(
        "--require-approval",
        action="store_true",
        help="Require human approval before executing the plan",
    )
    p.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve the plan (useful for CI)",
    )
    p.add_argument(
        "--edit-plan",
        action="store_true",
        help="Before approval, allow editing the todo list (comma-separated).",
    )
    p.add_argument(
        "--edit-default",
        default=None,
        help="Optional default edit value (comma-separated). Useful for quick testing.",
    )

    return p.parse_args()