# CHANGELOG

All notable changes to this repository are documented here.  
This project is intentionally structured as a **learning journal** for agentic architectures using LangGraph, LangChain, and modern LLM tooling.

---

## v0.4 – Supervisor–Worker Sub-Agents (Hub-and-Spoke)

### What changed
- Refactored the healthcheck into an explicit **supervisor–worker** architecture:
  - `supervisor_node` owns the task loop and sets `current`
  - `route_from_supervisor` routes to the correct worker or terminates
  - worker nodes (`weaviate_worker`, `postgres_worker`, `redis_worker`) perform only tool execution and return results
- Generalized retries across all checks using:
  - `attempts` tracking in state
  - `MAX_ATTEMPTS` cap
  - `BACKOFF_SECONDS` delay
- Preserved constrained LLM planning from v0.3, now used as an optional planner feeding the supervisor’s todo list
- Improved reporting to include per-check attempt counts and planning source

### Why this exists
This milestone demonstrates a production-aligned agent architecture:

> **Supervisors decide. Workers execute. State is the contract.**

This pattern sets the foundation for:
- human-in-the-loop approval gates
- tool-selection routing under constraints
- parallel worker fan-out
- persisted/resumable workflows

## v0.3 – Constrained LLM Planner with Observability
**Commit:** `feat(agentic): add constrained LLM planner with observability`

### What changed
- Introduced an **LLM-assisted planning step** to the healthcheck graph
- The LLM proposes a todo list *only* from an allowlisted set of checks
- All LLM output is **validated, sanitized, and gated** before execution
- Added **planner observability** into graph state:
  - raw LLM output
  - parsed JSON
  - validated todo
  - rejection / fallback reasons
- Added CLI flags:
  - `--llm-plan` to enable LLM planning
  - `--debug-plan` to surface raw / parsed planner output in reports

### Why this exists
This milestone demonstrates a core agentic principle:

> **LLMs decide _what_ to do, but never directly decide _what runs_.**

The system remains:
- deterministic by default
- safe under malformed or adversarial model output
- introspectable for debugging and audits

This pattern is intended as a foundation for:
- supervisor–worker architectures
- approval gates
- human-in-the-loop workflows

---

## v0.2 – Dynamic Planning, Routing, and Retries

### What changed
- Introduced a router node to dynamically dispatch checks
- Added per-check retry logic with capped attempts
- Execution order driven by a mutable `todo` list in state
- Report now reflects actual execution behavior (not static expectations)

### Why this exists
This milestone establishes:
- explicit control flow (no “magic agent loops”)
- observable state transitions
- production-aligned failure handling

---

## v0.1 – Deterministic Agentic Healthcheck

### What changed
- Built first LangGraph-based healthcheck
- Explicit nodes for Weaviate, Postgres, Redis
- Deterministic planner and final report
- No LLM usage

### Why this exists
This is the **baseline**: a fully understandable, fully deterministic agent graph.  
Everything after this builds on top of it.

---

## Design Philosophy (stable across versions)

- **No hidden autonomy**
- **State is explicit and inspectable**
- **LLMs are tools, not authorities**
- **Every agent decision must be observable**
- **Failure modes are intentional and boring**