from __future__ import annotations

from datetime import datetime

from .models import AccountEvent


def session_mark(
    account_id: str,
    timestamp: datetime,
    balance: float,
    realized_pnl: float,
    session_name: str,
    *,
    unrealized_pnl: float = 0.0,
    forced_exit: bool = False,
    mll_threshold: float = 0.0,
    daily_realized_pnl: float = 0.0,
) -> AccountEvent:
    """Create a deterministic session-boundary mark/flatten event.

    Native adapters can call this at exchange-session boundaries. The Phase D
    synthetic adapter supplies already-settled signals, so its normal replay
    does not invent additional forced exits.
    """

    marked_equity = balance + unrealized_pnl
    return AccountEvent(
        account_id=account_id,
        timestamp=timestamp,
        event_type="SESSION_FORCED_EXIT" if forced_exit else "SESSION_MARK",
        balance=balance,
        marked_equity=marked_equity,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        daily_realized_pnl=daily_realized_pnl,
        daily_unrealized_pnl=unrealized_pnl,
        fees=0,
        dlg_used=max(0, -daily_realized_pnl),
        mll_threshold=mll_threshold,
        remaining_mll_buffer=marked_equity - mll_threshold,
        reason=f"session boundary: {session_name}",
    )
