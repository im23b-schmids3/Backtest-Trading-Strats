from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from research_pipeline.cli import main
from research_pipeline.liquidity_sweep_mean_reversion.artifacts import validate_artifact_tree
from research_pipeline.liquidity_sweep_mean_reversion_v2.models import LSMRV2Config, TERMINAL_DISPOSITIONS, preregistered_candidates
from research_pipeline.liquidity_sweep_mean_reversion_v2.runner import materialize_lsmr_v2_strict_contract
from research_pipeline.liquidity_sweep_mean_reversion_v2.strategy import detect_setups, terminal_disposition, validate_setup_audit


def _bars(count=70, start=None):
    start = start or datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)
    return [{"timestamp": start + timedelta(minutes=5 * index), "open": "100", "high": "101", "low": "99", "close": "100", "volume": "10", "daily_vwap": "100"} for index in range(count)]


def _setup(index=24): return {"setup_id": "setup", "structure_id": "structure", "direction": "LONG", "reference": "99", "extreme": "99.0", "sweep_index": index, "event_history": [{"event": "PROPOSED_SETUP"}]}
def _reclaim(rows, index=24): rows[index].update(open="98", high="101", low="99", close="100", volume="20", daily_vwap="100")


def test_v2_registry_and_sealed_configuration():
    assert [(c.candidate_id, c.target_r_multiple) for c in preregistered_candidates()] == [("LSMR-V2-2P0R", Decimal("2.0")), ("LSMR-V2-2P5R", Decimal("2.5")), ("LSMR-V2-3P0R", Decimal("3.0"))]
    with pytest.raises(ValueError, match="invariant"): LSMRV2Config(reference_bars=12)


def test_v2_reference_penetration_floor_and_repeat_extreme_history():
    rows = _bars(); rows[24]["low"] = "98.5"
    setups, events = detect_setups(rows, LSMRV2Config())
    assert setups[0]["reference"] == "99" and setups[0]["event_history"][0]["event"] == "PROPOSED_SETUP"
    assert events[0]["setup_id"] == setups[0]["setup_id"]


@pytest.mark.parametrize(("change", "expected"), [(lambda rows: rows[24].update(open="99.9", close="100"), "CANDLE_REJECTED"), (lambda rows: rows[24].update(volume="14.9"), "VOLUME_REJECTED"), (lambda rows: rows[0].update(daily_vwap="99"), "REGIME_REJECTED"), (lambda rows: [row.update(daily_vwap="101") for row in rows], "VWAP_PROXIMITY_REJECTED")])
def test_v2_reclaim_filters_have_exact_terminal_dispositions(change, expected):
    rows = _bars(); _reclaim(rows); change(rows)
    assert terminal_disposition(_setup(), rows, LSMRV2Config())[0] == expected


def test_v2_session_context_requires_24_same_session_bars():
    rows = _bars(start=datetime(2023, 1, 1, 22, 0, tzinfo=timezone.utc)); _reclaim(rows)
    assert terminal_disposition(_setup(), rows, LSMRV2Config())[0] == "SESSION_CONTEXT_UNAVAILABLE"


def test_v2_stop_limits_next_open_and_two_bar_reclaim_window():
    rows = _bars(); rows[24]["close"] = rows[25]["close"] = "99"; _reclaim(rows, 26)
    assert terminal_disposition(_setup(), rows, LSMRV2Config())[0] == "RECLAIM_WINDOW_EXPIRED"
    rows = _bars(); _reclaim(rows); rows[25]["open"] = "100"
    disposition, trade = terminal_disposition(_setup(), rows, LSMRV2Config())
    assert disposition == "TRADE_EXECUTED" and trade and trade["entry_price"] == "100" and trade["trade_id"]


def test_v2_terminal_audit_is_exact_and_requires_trade_id():
    setup = _setup(); trade = {"setup_id": "setup", "trade_id": "trade"}
    validate_setup_audit([setup], [], [trade], [{"setup_id": "setup", "disposition": "TRADE_EXECUTED"}])
    with pytest.raises(AssertionError): validate_setup_audit([setup], [], [], [{"setup_id": "setup", "disposition": "TRADE_EXECUTED"}])
    assert "COOLDOWN_BLOCKED" in TERMINAL_DISPOSITIONS and "DUPLICATE_REFERENCE_SUPPRESSED" in TERMINAL_DISPOSITIONS


def test_v2_synthetic_cli_is_deterministic_and_never_executes_a_study(tmp_path, capsys):
    root = tmp_path / "repo"; spec = root / ".smithers/specs/liquidity-sweep-mean-reversion-v2-strict.md"; spec.parent.mkdir(parents=True)
    spec.write_text("\n".join(["LiquiditySweepMeanReversion.BTC_LONG_SHORT_V2_STRICT_SELECTION", "LSMR-V2-2P0R=2R LSMR-V2-2P5R=2.5R LSMR-V2-3P0R=3R", "SESSION_CONTEXT_UNAVAILABLE TRADE_EXECUTED"]), encoding="utf-8")
    result = materialize_lsmr_v2_strict_contract(repository_root=root, artifact_root=tmp_path / "runs")
    assert result["realStudyExecuted"] is False and validate_artifact_tree(result["artifactRoot"])["valid"]
    assert main(["lsmr-v2-strict-materialize", "--repository-root", str(root), "--artifact-root", str(tmp_path / "cli-runs")]) == 0
    assert '"realStudyExecuted": false' in capsys.readouterr().out
