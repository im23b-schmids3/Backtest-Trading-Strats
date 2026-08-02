from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyRow:
    strategy_id: str
    version: str
    current_phase: str
    approval_status: str
    terminal_status: str | None


@dataclass(frozen=True)
class TransitionRow:
    strategy_id: str
    old_state: str
    new_state: str
    timestamp: str
    reason: str

