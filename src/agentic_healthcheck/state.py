from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class HealthcheckState(TypedDict, total=False):
    # ----------------------------
    # Inputs
    # ----------------------------
    goal: str
    cli: Dict[str, Any]  # only / skip / llm_plan flags

    # ----------------------------
    # Planning / routing
    # ----------------------------
    plan: List[str]          # human-readable plan
    todo: List[str]          # machine plan
    current: str             # router-selected current task
    planning_source: str     # "llm" | "deterministic"

    # ----------------------------
    # LLM planner observability (3.2)
    # ----------------------------
    llm_plan_raw: str | None           # raw model output (text)
    llm_plan_parsed: Dict[str, Any] | None  # json.loads result
    llm_plan_validated: List[str] | None    # allowlisted todo we accepted
    llm_plan_error: str | None         # why plan was rejected

    # ----------------------------
    # Execution results
    # ----------------------------
    checks: Dict[str, Any]      # per-check results
    attempts: Dict[str, int]   # retry counters

    # ----------------------------
    # Final output
    # ----------------------------
    report: str