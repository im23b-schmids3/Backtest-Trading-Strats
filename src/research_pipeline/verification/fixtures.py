from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path

from .models import VerificationManifest


def make_fixture(output_dir: str | Path, strategy_id: str = "b5-fixture", strategy_version: str = "phase-b-1", kind: str = "correct") -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    data = {
        "trades": [{"trade_id": "T1", "signal_id": "S1", "strategy_id": strategy_id, "market": "TEST", "timeframe": "1h", "direction": "long", "entry_timestamp": "2026-01-01T10:00:00Z", "exit_timestamp": "2026-01-01T11:00:00Z", "entry_price": 100, "exit_price": 110, "quantity": 1, "contract_multiplier": 2, "gross_pnl": 20, "fees": 2, "slippage": 1, "net_pnl": 17, "exit_reason": "target", "data_source": "synthetic-fixture", "expected_gross_pnl": 20, "expected_fees": 2, "expected_slippage": 1}],
        "exit_legs": [{"trade_id": "T1", "leg_number": 1, "leg_type": "full", "leg_quantity": 1, "price": 110, "gross_pnl": 20, "fees": 2, "net_pnl": 18, "remaining_quantity": 0, "initial_quantity": 1}],
        "scaling_samples": [{"quantity": q, "pnl": 10 * q} for q in (1, 2, 5, 10)],
        "fee_reconciliation": [{"trade_id": "T1", "fees": 2, "expected_fees": 2, "basis": "per_side_fixed"}],
        "trade_counts": {"candidate_setups": 1, "order_versions": 2, "submitted_entries": 2, "filled_entries": 1, "accepted_entries": 1, "partially_closed_positions": 0, "fully_closed_positions": 1, "open_positions": 0, "cancelled_entries": 1, "session_forced_closures": 0, "total_trades": 1, "total_trades_definition": "completed positions"},
        "causality": {"lookahead_detected": False, "strategy_specific_checks": "PASS"},
        "session_boundary": {"expected_session_close_events": 1, "terminal_flatten_cluster": False, "timezone": "UTC", "dst_documented": True},
        "report_reconciliation": [{"metric": "total_net_pnl", "source_report": "summary", "source_rows": 1, "recomputed_value": 17, "reported_value": 17, "tolerance": 1e-9}],
        "data_sources": [{"provider": "fixture", "source_symbol": "TEST", "date_range": "2026-01-01/2026-01-01", "candle_count": 2, "native_or_proxy": "native", "synthetic_transformation": "none", "data_quality_status": "deterministic"}],
        "replay_hashes": ["fixture-replay-hash", "fixture-replay-hash"],
    }
    if kind == "missing-multiplier": data["trades"][0].update({"contract_multiplier": 1})
    elif kind == "duplicate-fee": data["trades"][0].update({"fees": 4, "net_pnl": 15}); data["fee_reconciliation"][0]["fees"] = 4
    elif kind == "partial-exit": data["exit_legs"][0].update({"leg_quantity": 2})
    elif kind == "scaling": data["scaling_samples"][1]["pnl"] = 21
    elif kind == "ambiguous-count": data["trade_counts"]["total_trades_definition"] = ""
    elif kind == "lookahead": data["causality"]["lookahead_detected"] = True
    elif kind == "terminal-flatten": data["session_boundary"]["terminal_flatten_cluster"] = True
    elif kind == "report-mismatch": data["report_reconciliation"][0]["reported_value"] = 19
    elif kind == "proxy-unlabeled": data["data_sources"][0].update({"native_or_proxy": "proxy", "synthetic_transformation": ""})
    elif kind == "nondeterministic": data["replay_hashes"][1] = "different-hash"
    elif kind == "missing-diagnostics":
        data = {"trades": data["trades"]}
    elif kind != "correct":
        raise ValueError(f"unknown fixture kind: {kind}")
    diagnostics = target / "diagnostics.json"
    diagnostics.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    manifest = VerificationManifest(strategy_id=strategy_id, strategy_version=strategy_version, verification_run_id=str(uuid.uuid4()), diagnostic_files=[str(diagnostics)])
    manifest.save(target / "manifest.yaml")
    return target / "manifest.yaml"
