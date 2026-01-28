from typing import TypedDict, Dict, List, Any


class HealthcheckState(TypedDict, total=False):
    goal: str
    plan: List[str]
    todo: List[str]        # NEW: remaining checks to run
    current: str           # NEW: which check we’re about to run
    checks: Dict[str, Any]
    report: str