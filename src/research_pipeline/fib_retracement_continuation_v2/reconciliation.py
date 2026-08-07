from datetime import timezone

from research_pipeline.fib_retracement_continuation_v1.reconciliation import reconcile as _v1_reconcile


def reconcile(setups, outcomes, orders, trades, opening_equity, *, final_equity=None,
              events=None, pending_after_cutoff=(), positions_after_cutoff=()):
    """V1 accounting reconciliation plus non-negotiable session invariants."""
    # V1 predates the V2 session terminal.  Map only for the frozen generic
    # lifecycle check; preserve the actual session disposition in output data.
    lifecycle_outcomes = [{**item, "disposition": "SESSION_OR_DATA_END"}
                          if item.get("disposition") == "SESSION_ENTRY_CUTOFF_2245" else item
                          for item in outcomes]
    base = _v1_reconcile(setups, lifecycle_outcomes, orders, trades, opening_equity,
                         final_equity=final_equity, events=events)
    forced = [leg for trade in trades for leg in trade.get("legs", [])
              if leg.get("reason") == "FORCED_SESSION_EXIT_2245"]
    cutoff_entries = [trade for trade in trades
                      if trade.get("entry_timestamp").astimezone(timezone.utc).time().hour == 22
                      and trade["entry_timestamp"].astimezone(timezone.utc).time().minute >= 45]
    overnight = [trade for trade in trades if any(
        leg.get("timestamp").date() != trade.get("entry_timestamp").date()
        for leg in trade.get("legs", []))]
    forced_ok = all(
        leg.get("timestamp").astimezone(timezone.utc).time().hour == 22
        and leg["timestamp"].astimezone(timezone.utc).time().minute == 45
        and leg.get("raw_price") == leg.get("cutoff_open")
        for leg in forced)
    checks = {
        **{key: value for key, value in base.items() if isinstance(value, bool)},
        "no_overnight_trade": not overnight,
        "no_position_after_cutoff": not positions_after_cutoff,
        "no_cutoff_entry": not cutoff_entries,
        "forced_exit_timestamp_and_open_price": forced_ok,
        "no_pending_after_cutoff": not pending_after_cutoff,
    }
    return {**base, **checks, "reconciles": all(checks.values()),
            "overnight_trade_count": len(overnight),
            "cutoff_entry_count": len(cutoff_entries),
            "forced_exit_count": len(forced)}
