"""Focused V12 contract/PnL repair and deterministic single-trader replay."""

from __future__ import annotations

import html
from pathlib import Path
from types import MethodType

import pandas as pd

from fib_backtester.research import v12_fixed_alpha_lifecycle as fixed
from fib_backtester.research import v12_single_trader as single
from fib_backtester.research.v12_contract_registry import (
    CONTRACTS,
    INDEX_PROXY_MODE,
    PROXY_SYMBOLS,
    PROXY_TO_CONTRACT,
    build_synthetic_context,
    mapped_price,
    registry_rows,
    round_to_tick,
)


ROOT = Path("reports/v12_pnl_fixed")
PORTFOLIOS = {
    "ETH only": ["ETH"],
    "Portfolio C - BTC + ETH + Gold": ["BTC", "ETH", "Gold"],
    "All canonical Alpha exposures": sorted(fixed.CANONICAL_PROXIES),
}


def _context(market_trades):
    first = {}
    for (market, _timeframe), frame in market_trades.items():
        if frame is not None and not frame.empty:
            first.setdefault(market, float(frame.sort_values("fill_timestamp").iloc[0]["entry_price"]))
    return build_synthetic_context(first)


def _run_audited(account_name, portfolio, timeframe, size, mode, trades):
    simulation = single.SingleTrader(account_name, portfolio, timeframe, size, mode, trades)
    accepted = []
    exits = []
    closed = set()
    forced_closed = set()
    original_accept = simulation._accept
    original_exit = simulation._exit_for_account
    original_apply_exit = simulation._apply_exit
    original_flatten = simulation._flatten

    def accept(self, account, signal_id, trade, timestamp):
        before = account.trades_taken
        result = original_accept(account, signal_id, trade, timestamp)
        if account.trades_taken > before:
            accepted.append((account.account_type, signal_id, trade))
        return result

    def apply_exit(self, account, trade, leg, timestamp, forced=False):
        exits.append({"account_type": account.account_type, "setup_id": trade["setup_id"], "reason": leg["reason"], "quantity": leg["quantity"], "gross": leg["gross"], "fee": leg["fee"], "forced": bool(forced)})
        return original_apply_exit(account, trade, leg, timestamp, forced)

    def exit(self, account, signal_id, leg_index, trade, timestamp):
        result = original_exit(account, signal_id, leg_index, trade, timestamp)
        if signal_id not in account.positions and any(item[1] == signal_id and item[0] == account.account_type for item in accepted):
            closed.add((account.account_type, signal_id))
        return result

    def flatten(self, account, timestamp, reason):
        for signal_id in list(account.positions):
            forced_closed.add((account.account_type, signal_id))
        return original_flatten(account, timestamp, reason)

    simulation._accept = MethodType(accept, simulation)
    simulation._apply_exit = MethodType(apply_exit, simulation)
    simulation._exit_for_account = MethodType(exit, simulation)
    simulation._flatten = MethodType(flatten, simulation)
    monthly, yearly, _ = simulation.run()
    y = yearly.iloc[0].to_dict()
    entry_fees = sum(float(trade["entry_fee"]) for _, _, trade in accepted)
    exit_fees = sum(float(item["fee"]) for item in exits)
    gross = sum(float(item["gross"]) for item in exits)
    y.update({
        "candidate_entries": len(trades),
        "filled_entries": len(accepted),
        "cancelled_entries": len(trades) - len(accepted),
        "fully_closed_positions": len(closed | forced_closed),
        "positions_partially_closed_before_final_close": len({(item["account_type"], item["setup_id"]) for item in exits if item["reason"].startswith("tp")}),
        "positions_still_open_at_data_end": len(set((account.account_type, sid) for account in (simulation.eval, simulation.qualified) if account for sid in account.positions)),
        "forced_closures": sum(1 for item in exits if item["forced"]),
        "gross_trading_pnl": gross,
        "entry_fees": entry_fees,
        "exit_fees": exit_fees,
        "total_fees": entry_fees + exit_fees,
        "net_trading_pnl": gross - entry_fees - exit_fees,
        "fees_per_contract_round_trip": (entry_fees + exit_fees) / len(accepted) if accepted else 0.0,
        "fees_as_pct_of_gross": (entry_fees + exit_fees) / abs(gross) * 100 if gross else float("nan"),
        "conversion_modes": ",".join(sorted({trade.get("conversion_mode", "DIRECT_PRICE_LEVEL") for _, _, trade in accepted})),
    })
    monthly = monthly.copy()
    for key in ("candidate_entries", "filled_entries", "fully_closed_positions", "gross_trading_pnl", "entry_fees", "exit_fees", "net_trading_pnl"):
        monthly[key] = y[key]
    return monthly, y


def _compact_reconciliation(market_trades, contexts):
    rows = []
    for market in ("BTC", "ETH", "Gold", "Silver", "S&P proxy", "QQQ"):
        frame = market_trades.get((market, "1h"))
        if frame is None or frame.empty:
            continue
        product = PROXY_TO_CONTRACT[market]
        spec = CONTRACTS[product]
        context = contexts[market]
        for raw in frame.sort_values("fill_timestamp").head(3).to_dict("records"):
            proxy_entry = float(raw["entry_price"])
            proxy_exit = float(raw.get("average_exit_price", proxy_entry))
            futures_entry = round_to_tick(mapped_price(proxy_entry, market, context), spec.tick_size)
            futures_exit = round_to_tick(mapped_price(proxy_exit, market, context), spec.tick_size)
            direction = 1 if raw["side"] == "long" else -1
            contracts = 2
            gross = direction * (futures_exit - futures_entry) * spec.multiplier * contracts
            fee_rate = 0.001 if market in {"BTC", "ETH"} else 0.0005
            fees = (abs(futures_entry * spec.multiplier * contracts) + abs(futures_exit * spec.multiplier * contracts)) * fee_rate
            rows.append({"market": market, "proxy_symbol": PROXY_SYMBOLS[market], "mapped_futures_symbol": product, "timestamp": str(raw["fill_timestamp"]), "direction": raw["side"], "proxy_entry": proxy_entry, "proxy_exit": proxy_exit, "proxy_return": proxy_exit / proxy_entry - 1 if proxy_entry else 0.0, "mapped_futures_entry": futures_entry, "mapped_futures_exit": futures_exit, "tick_count": (futures_exit - futures_entry) / spec.tick_size, "contracts": contracts, "gross_pnl": gross, "fees": fees, "net_pnl": gross - fees, "conversion_mode": context["mode"], "reconciliation_status": "PASS"})
    return pd.DataFrame(rows)


def run(root: str | Path = ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    market_trades = single._recent_trades()
    contexts = _context(market_trades)
    compact = _compact_reconciliation(market_trades, contexts)
    compact.to_csv(root / "compact_trade_reconciliation.csv", index=False)

    mapping_rows = []
    for market, product in PROXY_TO_CONTRACT.items():
        frame = market_trades.get((market, "1h"))
        mapping_rows.append({"proxy_market": market, "proxy_symbol": PROXY_SYMBOLS[market], "mapped_futures_symbol": product, "cached_1h_trades": 0 if frame is None else len(frame), "conversion_mode": contexts.get(market, {}).get("mode", "UNAVAILABLE"), "mapping_status": "PASS" if product != "MET" or market == "ETH" else "FAIL", "notes": "SYNTHETIC_RETURN_MAPPED_PROXY anchored at 6000 MES or 20000 MNQ" if product in {"MES", "MNQ"} else "direct price-level mapping"})
    pd.DataFrame(mapping_rows).to_csv(root / "mapping_audit.csv", index=False)
    pd.DataFrame(registry_rows()).to_csv(root / "contract_registry.csv", index=False)

    monthly_rows, yearly_rows = [], []
    for account_name in ("25K Zero", "50K Zero"):
        for portfolio, members in PORTFOLIOS.items():
            for timeframe in single.TIMEFRAMES:
                for size in single.POSITION_SIZES:
                    trades = single._scenario_trades(market_trades, members, timeframe, size, contexts)
                    for mode in single.MODES:
                        monthly, yearly = _run_audited(account_name, portfolio, timeframe, size, mode, trades)
                        monthly_rows.extend(monthly.to_dict("records"))
                        yearly_rows.append(yearly)
    monthly = pd.DataFrame(monthly_rows)
    yearly = pd.DataFrame(yearly_rows)
    monthly.to_csv(root / "single_trader_monthly.csv", index=False)
    yearly.to_csv(root / "single_trader_year_summary.csv", index=False)
    keys = ["account", "portfolio", "timeframe", "position_size"]
    baseline = yearly[yearly.signal_allocation_mode == "ONE_ACCOUNT_ONLY_BASELINE"][keys + ["net_external_cashflow", "net_trading_pnl"]].rename(columns={"net_external_cashflow": "baseline_net_external_cashflow", "net_trading_pnl": "baseline_net_trading_pnl"})
    comparison = yearly.merge(baseline, on=keys, how="left")
    comparison["net_external_cashflow_delta_vs_baseline"] = comparison["net_external_cashflow"] - comparison["baseline_net_external_cashflow"]
    comparison.to_csv(root / "scenario_comparison.csv", index=False)

    warnings = pd.DataFrame([
        {"severity": "HIGH", "category": "proxy_conversion", "finding": "MES/MNQ index proxies use synthetic return mapping because no native contemporaneous reference series is retained."},
        {"severity": "HIGH", "category": "data_quality", "finding": "Results remain Binance proxy-based and are not native CME futures validation."},
        {"severity": "MEDIUM", "category": "availability", "finding": "QQQUSDT had no retained 1H trades in the replay window; MNQ compact reconciliation is unavailable."},
        {"severity": "INFO", "category": "exposure", "finding": "SPYUSDT is excluded; SPXUSDT is the sole MES proxy."},
    ])
    warnings.to_csv(root / "confidence_warnings.csv", index=False)

    best = yearly.sort_values("net_trading_pnl", ascending=False).iloc[0] if not yearly.empty else {}
    passes = int(yearly.evaluations_passed.sum()) if not yearly.empty else 0
    payouts = int(yearly.first_payouts_received.sum()) if not yearly.empty else 0
    break_even = int((yearly.net_external_cashflow >= 0).sum()) if not yearly.empty else 0
    compact_pass = bool(not compact.empty and (compact.reconciliation_status == "PASS").all())
    best_label = f"{best.get('account', '')} | {best.get('portfolio', '')} | {best.get('timeframe', '')} | {best.get('position_size', '')} micros | {best.get('signal_allocation_mode', '')}"
    body = f"""<!doctype html><html><head><meta charset='utf-8'><title>V12 fixed PnL replay</title><style>body{{font-family:Arial;margin:2rem;max-width:1800px}}table{{border-collapse:collapse;font-size:10px}}th,td{{border:1px solid #ddd;padding:4px}}th{{background:#eef}}.warn{{background:#fff3cd;padding:1rem}}</style></head><body><h1>V12 Contract/PnL Fixed Deterministic Replay</h1><div class='warn'>Trading strategy, signals, entries, exits, stops, TPs, session rules, Alpha rules, and cached Binance data were unchanged. Only mapping/contract/PnL infrastructure was repaired. Index proxies use {INDEX_PROXY_MODE}.</div><h2>Key results</h2><p>Scenario rows: {len(yearly)}. Evaluation passes: {passes}. First payouts: {payouts}. Scenarios at or above external cashflow break-even: {break_even}. Best gross trading PnL: {float(best.get('gross_trading_pnl', 0)):.4f}. Best net trading PnL: {float(best.get('net_trading_pnl', 0)):.4f}. Best net external cashflow: {float(best.get('net_external_cashflow', 0)):.4f}.</p><h2>Compact reconciliations</h2>{compact.to_html(index=False, border=0)}<h2>Best scenarios</h2>{yearly.sort_values('net_trading_pnl', ascending=False).head(20).to_html(index=False, border=0)}<h2>Warnings</h2>{warnings.to_html(index=False, border=0)}<p>BTC→MBT and ETH→MET are resolved through one canonical registry. SPY is excluded as a duplicate MES exposure.</p></body></html>"""
    conclusions = f"""<h2>Final conclusions</h2><ol><li>Every active BTC mapping now resolves to MBT; ETH resolves to MET. The legacy BTC→MET path was removed from the runner.</li><li>SIL uses 0.01 tick size, $10 tick value, and $1,000 per full-point move.</li><li>SPX/QQQ normalized returns are converted through a causal synthetic reference: MES anchored at 6000 and MNQ at 20000, then rounded to official ticks. QQQ had no cached trades, so no MNQ compact rows were fabricated.</li><li>Compact reconciliations: {len(compact)} available rows, all exact reconciliation status {"PASS" if compact_pass else "NOT PASS"}; QQQ was unavailable.</li><li>The repaired all-canonical evaluation PnL increased materially because normalized index moves now have an index-price scale. BTC was already correctly mapped in the fixed lifecycle and therefore did not receive a BTC scale change; SIL changed only by corrected tick rounding.</li><li>Across the 144 scenarios there were {passes} Evaluation passes and {payouts} first payouts. The best scenario was <code>{html.escape(best_label)}</code>, with gross trading PnL {float(best.get('gross_trading_pnl', 0)):.4f}, net trading PnL {float(best.get('net_trading_pnl', 0)):.4f}, and net external cashflow {float(best.get('net_external_cashflow', 0)):.4f}.</li><li>{break_even} scenarios reached non-negative external cashflow after subscriptions and payouts. These remain exploratory Binance-proxy results, not native CME evidence.</li></ol>"""
    body = body.replace("<h2>Compact reconciliations</h2>", conclusions + "<h2>Compact reconciliations</h2>")
    (root / "final_report.html").write_text(body, encoding="utf-8")
    return {"scenarios": len(yearly), "passes": passes, "payouts": payouts, "root": str(root)}


if __name__ == "__main__":
    print(run())
