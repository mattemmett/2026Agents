from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

import requests
import psycopg
import redis

from .state import HealthcheckState
from .cli import parse_args
from src.llm import get_llm


# ----------------------------
# Planning (deterministic + optional LLM)
# ----------------------------

ALLOWED_CHECKS = ["weaviate", "postgres", "redis"]


def _parse_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [c.strip() for c in value.split(",") if c.strip()]


def llm_propose_todo(goal: str, allowed: List[str]) -> Dict[str, Any]:
    """
    Ask the LLM for a plan, but ONLY accept allowlisted output.

    Returns an "observability bundle" dict:
      - ok: bool (whether we accepted a validated todo)
      - raw: str (raw model text)
      - parsed: dict|None (json.loads result if it worked)
      - validated_todo: list[str]|None (the allowlisted todo we accepted)
      - error: str|None (why we rejected / failed)
    """
    llm = get_llm(temperature=0.0)

    system = SystemMessage(
        content=(
            "You are a planning assistant for a devcontainer healthcheck.\n"
            "You MUST respond with a single JSON object and NOTHING else.\n"
            'Schema: {"todo": ["weaviate"|"postgres"|"redis", ...]}\n'
            f"Allowed values: {allowed}\n"
            "Rules:\n"
            "- Output valid JSON (no markdown, no code fences).\n"
            "- Only include allowed values.\n"
            "- Use at most 3 items.\n"
        )
    )

    human = HumanMessage(
        content=(
            f"Goal: {goal}\n"
            "Propose an efficient todo list of checks to run."
        )
    )

    resp = llm.invoke([system, human])
    raw = (resp.content or "").strip()

    bundle: Dict[str, Any] = {
        "ok": False,
        "raw": raw,
        "parsed": None,
        "validated_todo": None,
        "error": None,
    }

    # Parse JSON
    try:
        obj = json.loads(raw)
        bundle["parsed"] = obj
    except Exception as e:
        bundle["error"] = f"json_parse_failed: {e}"
        return bundle

    # Validate schema
    todo = bundle["parsed"].get("todo") if isinstance(bundle["parsed"], dict) else None
    if not isinstance(todo, list):
        bundle["error"] = "schema_invalid: todo must be a list"
        return bundle

    cleaned: List[str] = []
    for item in todo:
        if not isinstance(item, str):
            bundle["error"] = "schema_invalid: todo items must be strings"
            return bundle
        item = item.strip()
        if item not in allowed:
            bundle["error"] = f"not_allowlisted: {item}"
            return bundle
        if item not in cleaned:
            cleaned.append(item)

    if len(cleaned) > 3:
        bundle["error"] = "too_long: max 3 items"
        return bundle

    if not cleaned:
        bundle["error"] = "empty_plan: todo must not be empty"
        return bundle

    bundle["ok"] = True
    bundle["validated_todo"] = cleaned
    return bundle


def plan_node(state: HealthcheckState) -> HealthcheckState:
    goal = state.get("goal", "Check dev services")

    cli = state.get("cli", {}) or {}
    only = cli.get("only")
    skip = cli.get("skip")
    llm_plan = bool(cli.get("llm_plan", False))

    only_list = _parse_csv(only)
    skip_list = _parse_csv(skip)

    # Start from allowlist
    plan = list(ALLOWED_CHECKS)

    # Manual constraints win over LLM planning
    if only_list:
        plan = [c for c in only_list if c in ALLOWED_CHECKS]

    if skip_list:
        skip_set = set(skip_list)
        plan = [c for c in plan if c not in skip_set]

    planning_source = "deterministic"

    # Observability fields (default)
    llm_plan_raw: Optional[str] = None
    llm_plan_parsed: Optional[dict] = None
    llm_plan_validated: Optional[List[str]] = None
    llm_plan_error: Optional[str] = None

    # If --llm-plan and no manual constraints, let LLM propose todo
    if llm_plan and not only_list and not skip_list:
        bundle = llm_propose_todo(goal, ALLOWED_CHECKS)
        llm_plan_raw = bundle.get("raw")
        llm_plan_parsed = bundle.get("parsed")
        llm_plan_validated = bundle.get("validated_todo")
        llm_plan_error = bundle.get("error")

        if bundle.get("ok") and llm_plan_validated:
            plan = llm_plan_validated
            planning_source = "llm"
        else:
            planning_source = "deterministic"

    return {
        "goal": goal,
        "plan": [f"Check {c.capitalize()}" for c in plan] + ["Summarize results"],
        "todo": plan,
        "planning_source": planning_source,
        "llm_plan_raw": llm_plan_raw,
        "llm_plan_parsed": llm_plan_parsed,
        "llm_plan_validated": llm_plan_validated,
        "llm_plan_error": llm_plan_error,
    }

def approval_gate_node(state: HealthcheckState) -> HealthcheckState:
    """
    Human-in-the-loop approval gate.
    Runs AFTER planning, BEFORE any execution.

    Behavior:
      - If approval not required: auto-approve.
      - If auto-approve set: approve and continue.
      - Else:
          - optionally allow editing the todo (if --edit-plan)
          - then prompt for approval
    """
    cli = state.get("cli", {}) or {}
    require = bool(cli.get("require_approval", False))
    auto = bool(cli.get("auto_approve", False))
    edit_plan = bool(cli.get("edit_plan", False))
    edit_default = cli.get("edit_default")

    # Default: no approval needed
    if not require:
        return {
            "approval_required": False,
            "approved": True,
            "approval_reason": "not_required",
        }

    # Approval required, but auto-approved (CI / non-interactive)
    if auto:
        return {
            "approval_required": True,
            "approved": True,
            "approval_reason": "auto_approve",
        }

    # Interactive flow (optional edit, then approve)
    source = state.get("planning_source", "unknown")
    todo = list(state.get("todo", []))
    plan = list(state.get("plan", []))

    edited = False
    edit_error = None

    if edit_plan:
        print("\n--- EDIT PLAN (optional) ---")
        print(f"Allowed checks: {', '.join(ALLOWED_CHECKS)}")
        print(f"Current todo: {', '.join(todo) if todo else '(empty)'}")

        raw = input("Edit todo (comma-separated) or press Enter to keep: ").strip()

        # Optional: if user hits Enter, but you provided --edit-default (testing)
        if not raw and edit_default:
            raw = edit_default.strip()

        if raw:
            proposed = [x.strip() for x in raw.split(",") if x.strip()]
            cleaned: list[str] = []
            invalid: list[str] = []

            for item in proposed:
                if item not in ALLOWED_CHECKS:
                    invalid.append(item)
                    continue
                if item not in cleaned:
                    cleaned.append(item)

            if cleaned:
                todo = cleaned
                plan = [f"Check {c.capitalize()}" for c in todo] + ["Summarize results"]
                edited = True
                if invalid:
                    edit_error = f"ignored_invalid={invalid}"
            else:
                # If they entered only invalid values, keep existing todo
                edit_error = f"edit_rejected_all_invalid={invalid or proposed}"

    print("\n--- HUMAN APPROVAL REQUIRED ---")
    print(f"Planning source: {source}")
    if edited:
        print("Plan was edited by human before approval.")
    if edit_error:
        print(f"Edit note: {edit_error}")

    print("Proposed plan:")
    for i, step in enumerate(plan, 1):
        print(f"  {i}. {step}")

    resp = input("Approve this plan? [y/N]: ").strip().lower()
    approved = resp in ("y", "yes")

    reason = "user_approved" if approved else "user_denied"
    if edited and approved:
        reason = "user_edited_then_approved"
    elif edited and not approved:
        reason = "user_edited_then_denied"

    out: HealthcheckState = {
        "approval_required": True,
        "approved": approved,
        "approval_reason": reason,
        # If edited, persist the edited plan into state so execution matches what was approved
        "todo": todo,
        "plan": plan,
    }

    # Optional: keep this for report/debug visibility
    if edit_error:
        out["approval_edit_note"] = edit_error  # you'll add this to state.py if you want to type it

    return out


def route_from_approval(state: HealthcheckState) -> str:
    return "approved" if state.get("approved") else "denied"


# ----------------------------
# Supervisor + Worker orchestration
# ----------------------------

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 0.25


def bump_attempts(state: HealthcheckState, name: str) -> Dict[str, int]:
    attempts = dict(state.get("attempts", {}))
    attempts[name] = attempts.get(name, 0) + 1
    return attempts


def merge_check(state: HealthcheckState, name: str, result: dict, attempts: Dict[str, int]) -> HealthcheckState:
    checks = dict(state.get("checks", {}))
    checks[name] = result
    return {"checks": checks, "attempts": attempts}


def supervisor_node(state: HealthcheckState) -> HealthcheckState:
    """
    Supervisor owns the task loop:
      - picks next task from todo
      - sets current
      - no direct tool calls
    """
    todo = list(state.get("todo", []))
    if not todo:
        return {"current": ""}

    current = todo.pop(0)
    return {"current": current, "todo": todo}


def route_from_supervisor(state: HealthcheckState) -> str:
    """
    Supervisor decides where to go next:
      - to a worker ("weaviate"/"postgres"/"redis")
      - or to report ("done")

    Also handles retry decision if a worker failed and attempts remain.
    """
    current = state.get("current", "")
    if current not in ("weaviate", "postgres", "redis"):
        return "done"

    checks = state.get("checks", {})
    attempts = state.get("attempts", {})

    result = checks.get(current)

    # If we already ran it and it failed, retry while under cap.
    if result and not result.get("ok", False):
        if attempts.get(current, 0) < MAX_ATTEMPTS:
            time.sleep(BACKOFF_SECONDS)
            return current

    return current


# ----------------------------
# Worker nodes (pure tool execution)
# ----------------------------

def weaviate_worker(state: HealthcheckState) -> HealthcheckState:
    attempts = bump_attempts(state, "weaviate")
    try:
        resp = requests.get("http://weaviate:8080/v1/meta", timeout=3)
        resp.raise_for_status()
        meta = resp.json()
        result = {"ok": True, "meta": meta}
    except Exception as e:
        result = {"ok": False, "error": str(e)}
    return merge_check(state, "weaviate", result, attempts)


def postgres_worker(state: HealthcheckState) -> HealthcheckState:
    attempts = bump_attempts(state, "postgres")
    try:
        with psycopg.connect(
            host="postgres",
            dbname="agentic",
            user="agentic",
            password="agentic",
            connect_timeout=3,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("select 1;")
                cur.fetchone()
        result = {"ok": True}
    except Exception as e:
        result = {"ok": False, "error": str(e)}
    return merge_check(state, "postgres", result, attempts)


def redis_worker(state: HealthcheckState) -> HealthcheckState:
    attempts = bump_attempts(state, "redis")
    try:
        r = redis.Redis(host="redis", port=6379, socket_connect_timeout=2, socket_timeout=2)
        ok = bool(r.ping())
        result = {"ok": ok}
    except Exception as e:
        result = {"ok": False, "error": str(e)}
    return merge_check(state, "redis", result, attempts)


# ----------------------------
# Reporting
# ----------------------------

def report_node(state: HealthcheckState) -> HealthcheckState:
    plan = state.get("plan", [])
    report_lines = [f"Goal: {state.get('goal', '(none)')}"]

    cli = state.get("cli", {}) or {}
    debug_plan = bool(cli.get("debug_plan", False))

    source = state.get("planning_source", "unknown")
    report_lines.append(f"Planning source: {source}")

    # HITL status    
    if state.get("approval_required"):
        report_lines.append(
            f"Approval: {state.get('approval_reason')} (approved={state.get('approved')})"
        )
    elif state.get("approved") is not None:
        report_lines.append(f"Approval: {state.get('approval_reason')}")

    # If approved, show final todo
    approved = state.get("approved")
    todo = state.get("todo")
    if approved and todo is not None:
        report_lines.append(f"Final approved todo: {todo}")
    
    if source == "llm":
        report_lines.append(f"LLM validated todo: {state.get('llm_plan_validated')}")
    else:
        if state.get("llm_plan_raw") is not None:
            report_lines.append(f"LLM plan rejected: {state.get('llm_plan_error')}")

    if debug_plan and state.get("llm_plan_raw") is not None:
        raw = state.get("llm_plan_raw") or ""
        raw_trunc = raw if len(raw) <= 500 else raw[:500] + "…(truncated)"
        report_lines.append("LLM raw output:")
        report_lines.append(f"  {raw_trunc}")

        parsed = state.get("llm_plan_parsed")
        if parsed is not None:
            pretty = json.dumps(parsed, indent=2, sort_keys=True)
            pretty_trunc = pretty if len(pretty) <= 1200 else pretty[:1200] + "\n…(truncated)"
            report_lines.append("LLM parsed JSON:")
            report_lines.append(pretty_trunc)

    report_lines.append("Plan:")
    report_lines += [f"  {i}. {step}" for i, step in enumerate(plan, 1)]

    checks = state.get("checks", {})
    attempts = state.get("attempts", {})
    report_lines.append("Checks:")

    def fmt(name: str) -> str:
        if name not in checks:
            return "skipped"
        data = checks.get(name, {})
        a = attempts.get(name, 1)
        if data.get("ok"):
            return f"ok (attempts={a})"
        err = data.get("error", "no error captured")
        return f"failed ({err}) (attempts={a})"

    for name in ("weaviate", "postgres", "redis"):
        report_lines.append(f"  {name}: {fmt(name)}")

    report_lines.append(f"Remaining todo: {state.get('todo', [])}")
    return {"report": "\n".join(report_lines)}


# ----------------------------
# Graph build
# ----------------------------

def build_graph():
    g = StateGraph(HealthcheckState)

    # --- Planning ---
    g.add_node("plan", plan_node)

    # --- Human-in-the-loop approval ---
    g.add_node("approval_gate", approval_gate_node)

    # Supervisor/worker structure
    g.add_node("supervisor", supervisor_node)
    g.add_node("weaviate", weaviate_worker)
    g.add_node("postgres", postgres_worker)
    g.add_node("redis", redis_worker)

    #-- Reporting ---
    g.add_node("report", report_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "approval_gate")

    # Approval gate routes either to supervisor or straight to report
    g.add_conditional_edges(
        "approval_gate",
        route_from_approval,
        {
            "approved": "supervisor",
            "denied": "report",
        },
    )

    # Supervisor routes to workers or report
    g.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "weaviate": "weaviate",
            "postgres": "postgres",
            "redis": "redis",
            "done": "report",
        },
    )

    # Workers report back to supervisor only
    g.add_edge("weaviate", "supervisor")
    g.add_edge("postgres", "supervisor")
    g.add_edge("redis", "supervisor")

    g.add_edge("report", END)

    return g.compile()


# ----------------------------
# Entrypoint
# ----------------------------

if __name__ == "__main__":
    args = parse_args()
    graph = build_graph()

    initial_state: HealthcheckState = {
        "goal": "Verify devcontainer services",
        "cli": {
            "only": args.only,
            "skip": args.skip,
            "llm_plan": args.llm_plan,
            "debug_plan": getattr(args, "debug_plan", False),
            "require_approval": getattr(args, "require_approval", False),
            "auto_approve": getattr(args, "auto_approve", False),
            "edit_plan": getattr(args, "edit_plan", False),
            "edit_default": getattr(args, "edit_default", None),
        },
        "attempts": {},
        "checks": {},
        "approval_required": False,
        "approved": None,
        "approval_reason": None,
    }

    result = graph.invoke(initial_state)
    print(result["report"])