from __future__ import annotations

from langgraph.graph import StateGraph, START, END
import requests
import psycopg
import redis

from .state import HealthcheckState
from .cli import parse_args


# ----------------------------
# Planning + routing
# ----------------------------

def plan_node(state: HealthcheckState) -> HealthcheckState:
    goal = state.get("goal", "Check dev services")

    all_checks = ["weaviate", "postgres", "redis"]

    cli = state.get("cli", {}) or {}
    only = cli.get("only")
    skip = cli.get("skip")

    if only:
        plan = [c.strip() for c in only.split(",") if c.strip() in all_checks]
    else:
        plan = list(all_checks)

    if skip:
        skip_set = {c.strip() for c in skip.split(",") if c.strip()}
        plan = [c for c in plan if c not in skip_set]

    return {
        "goal": goal,
        "plan": [f"Check {c.capitalize()}" for c in plan] + ["Summarize results"],
        "todo": plan,
    }


def router_node(state: HealthcheckState) -> HealthcheckState:
    todo = list(state.get("todo", []))
    if not todo:
        return {"current": ""}  # signals we’re done

    current = todo.pop(0)
    return {"current": current, "todo": todo}


def route_from_router(state: HealthcheckState) -> str:
    current = state.get("current", "")
    if current in ("weaviate", "postgres", "redis"):
        return current
    return "done"


# ----------------------------
# Check nodes (tools)
# ----------------------------

def weaviate_check_node(state: HealthcheckState) -> HealthcheckState:
    resp = requests.get("http://weaviate:8080/v1/meta", timeout=3)
    meta = resp.json()

    checks = dict(state.get("checks", {}))
    checks["weaviate"] = {"ok": True, "meta": meta}
    return {"checks": checks}


def postgres_check_node(state: HealthcheckState) -> HealthcheckState:
    try:
        with psycopg.connect(
            host="postgres",
            dbname="agentic",
            user="agentic",
            password="agentic",
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("select 1;")
                cur.fetchone()
        result = {"ok": True}
    except Exception as e:
        result = {"ok": False, "error": str(e)}

    checks = dict(state.get("checks", {}))
    checks["postgres"] = result
    return {"checks": checks}


def redis_check_node(state: HealthcheckState) -> HealthcheckState:
    try:
        r = redis.Redis(host="redis", port=6379)
        ok = bool(r.ping())
        result = {"ok": ok}
    except Exception as e:
        result = {"ok": False, "error": str(e)}

    checks = dict(state.get("checks", {}))
    checks["redis"] = result
    return {"checks": checks}


# ----------------------------
# Reporting
# ----------------------------

def report_node(state: HealthcheckState) -> HealthcheckState:
    plan = state.get("plan", [])
    report_lines = [f"Goal: {state.get('goal', '(none)')}", "Plan:"]
    report_lines += [f"  {i}. {step}" for i, step in enumerate(plan, 1)]

    checks = state.get("checks", {})
    report_lines.append("Checks:")

    # weaviate
    if "weaviate" not in checks:
        report_lines.append("  weaviate: skipped")
    else:
        report_lines.append("  weaviate: ok")

    # postgres
    pg = checks.get("postgres")
    if pg is None:
        report_lines.append("  postgres: skipped")
    elif pg.get("ok"):
        report_lines.append("  postgres: ok")
    else:
        report_lines.append(f"  postgres: failed ({pg.get('error', 'no error captured')})")

    # redis (keep special error capture)
    rd = checks.get("redis")
    if rd is None:
        report_lines.append("  redis: skipped")
    elif rd.get("ok"):
        report_lines.append("  redis: ok")
    else:
        report_lines.append(f"  redis: failed ({rd.get('error', 'no error captured')})")

    report_lines.append(f"Remaining todo: {state.get('todo', [])}")

    return {"report": "\n".join(report_lines)}


# ----------------------------
# Graph build
# ----------------------------

def build_graph():
    g = StateGraph(HealthcheckState)

    g.add_node("plan", plan_node)
    g.add_node("router", router_node)

    g.add_node("weaviate", weaviate_check_node)
    g.add_node("postgres", postgres_check_node)
    g.add_node("redis", redis_check_node)

    g.add_node("report", report_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "router")

    g.add_conditional_edges(
        "router",
        route_from_router,
        {
            "weaviate": "weaviate",
            "postgres": "postgres",
            "redis": "redis",
            "done": "report",
        },
    )

    g.add_edge("weaviate", "router")
    g.add_edge("postgres", "router")
    g.add_edge("redis", "router")

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
        },
    }

    result = graph.invoke(initial_state)
    print(result["report"])