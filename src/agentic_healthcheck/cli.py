from __future__ import annotations

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Agentic devcontainer healthcheck")
    p.add_argument("--only", default=None, help="Comma-separated checks to run (weaviate,postgres,redis)")
    p.add_argument("--skip", default=None, help="Comma-separated checks to skip (weaviate,postgres,redis)")
    p.add_argument("--llm-plan", action="store_true", help="Use a constrained LLM to propose the todo order/subset")
    p.add_argument("--debug-plan", action="store_true", help="Print raw/parsed LLM planner output in the report")
    return p.parse_args()