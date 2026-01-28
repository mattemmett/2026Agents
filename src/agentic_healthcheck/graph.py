from langgraph.graph import StateGraph, START, END
from langsmith import trace
import requests, psycopg, redis

from .state import HealthcheckState
from .cli import parse_args




def plan_node(state: HealthcheckState) -> HealthcheckState:
    # A node reads the incoming state...
    goal = state.get("goal", "Check dev services")
    plan = ["weaviate", "postgres", "redis"]  # machine-friendly names


    # ...and returns only the fields it wants to update.
    return {
        "goal": goal,
        "plan": [
            "Check Weaviate",
            "Check Postgres",
            "Check Redis",
            "Summarize results",
        ],
        "todo": plan,
    }

def router_node(state: HealthcheckState) -> HealthcheckState:
    todo = list(state.get("todo", []))
    if not todo:
        return {"current": ""}  # signals we’re done

    current = todo.pop(0)
    return {"current": current, "todo": todo}

def weaviate_check_node(state: HealthcheckState) -> HealthcheckState:
    # Call Weaviate's /meta endpoint (simple, always JSON)
    resp = requests.get("http://weaviate:8080/v1/meta", timeout=3)
    meta = resp.json()

    # Merge into checks (don't overwrite other checks we'll add later)
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

def report_node(state: HealthcheckState) -> HealthcheckState:
    # Build a simple human-readable report from the current state.
    plan = state.get("plan", [])
    report_lines = [f"Goal: {state.get('goal', '(none)')}", "Plan:"]
    report_lines += [f"  {i}. {step}" for i, step in enumerate(plan, 1)]

    # 👇 NEW PART: read what previous nodes added to state
    checks = state.get("checks", {})
    report_lines.append("Checks:")
    report_lines.append(
        f"  weaviate: {'present' if 'weaviate' in checks else 'missing'}"
    )
    report_lines.append(
    f"  postgres: {'ok' if checks.get('postgres', {}).get('ok') else 'failed'}"
    )
    rd = checks.get("redis", {})


    if rd.get("ok"):
        report_lines.append("  redis: ok")
    else:
        report_lines.append(f"  redis: failed ({rd.get('error', 'no error captured')})")
    
    # Show that routing consumed the plan dynamically
    report_lines.append(f"Remaining todo: {state.get('todo', [])}")

    return {"report": "\n".join(report_lines)}

def route_from_router(state: HealthcheckState) -> str:
    current = state.get("current", "")
    if current in ("weaviate", "postgres", "redis"):
        return current
    return "done"


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

    # Conditional routing from router → specific check OR done
    g.add_conditional_edges(
        "router",
        route_from_router,
        {
            "weaviate": "weaviate",
            "postgres": "postgres",
            "redis": "redis",
            "done": "report"
        }
    )

    # After each check, go back to router (loop)
    g.add_edge("weaviate", "router")
    g.add_edge("postgres", "router")
    g.add_edge("redis", "router")
    
    g.add_edge("report", END)

    return g.compile()


if __name__ == "__main__":
    args = parse_args()
    graph = build_graph()
    final_state = graph.invoke({"goal": "Verify devcontainer services"})
    print(final_state["report"])