from typing import TypedDict, Dict, List, Any, Optional


class HealthcheckState(TypedDict, total=False):
    goal: str
    plan: List[str]
    todo: List[str]
    current: str
    checks: Dict[str, Any]
    report: str
    cli: Dict[str, Optional[str]]
    attempts: dict[str, int]
