from research_pipeline.cme_orderflow_absorption_v1.oos_backtest_runner import COUNTERS, EXPECTED_RECORDS, State, _score, completed_state, load_development_calibration, plus_only, prices, reconcile, size_trade
import json
from pathlib import Path

ROOT=Path(__file__).parents[2]
def contract(): return json.loads((ROOT/'docs/research_pipeline/cme_orderflow_absorption_v1/backtest-contract.json').read_text())
def test_plus_and_response_exclusion():
    r={"interaction_id":"x","interaction_end":1,"level":"PRIOR_RTH_POC","absorption_score":.8,"replenishment_score":.8,"direction":"BUYER_ABSORPTION"}
    assert plus_only(contract(),r)
    try: plus_only(contract(),dict(r,response_5s=99))
    except RuntimeError: pass
    else: assert False
def test_exact_prices_and_integer_ledger():
    p=prices("BUYER_ABSORPTION",5000,5000.25,4995,5005); s=size_trade(p)
    assert p["entry"]==5000.5 and p["stop"]==4993.75 and p["stop_exit"]==4993.5
    assert s["contracts"]==0 and s["one_contract_price_risk_usd"]==350

def complete_state(**changes):
    state=State(source_index=EXPECTED_RECORDS)
    state.counters.update({"dbn_records_seen":EXPECTED_RECORDS,"rth_sessions_processed":5})
    state.actual_rth_dates_processed=["2026-08-03","2026-08-04","2026-08-05","2026-08-06","2026-08-07"]
    state.counters.update(changes)
    return state

def test_complete_requires_all_record_and_session_counters():
    state=complete_state(dbn_records_seen=EXPECTED_RECORDS-1)
    assert not completed_state(state)
    state=complete_state(rth_sessions_processed=4)
    assert not completed_state(state)

def test_valid_zero_plus_reconciles_without_default_empty_ledger_claim():
    state=complete_state(plus_eligible=0,audit_rows=0,entered_trades=0,closed_trades=0)
    reconcile(state)

def test_plus_requires_a_real_audit_row_not_default_empty_ledgers():
    state=complete_state(plus_eligible=1,audit_rows=0)
    try: reconcile(state)
    except RuntimeError: pass
    else: assert False
    state.audit=[{"interaction_id":"x","outcome":"LATENCY_NOT_REACHED"}]
    state.counters["audit_rows"]=1
    reconcile(state)

def test_required_counters_and_frozen_literals_and_no_july_execution_wiring():
    assert {"snapshot_records_seen","ordinary_records_seen","high_absorption","strong_replenishment","plus_eligible","latency_not_reached"} <= set(COUNTERS)
    source=(ROOT/'src/research_pipeline/cme_orderflow_absorption_v1/oos_backtest_runner.py').read_text()
    assert 'absorption_p95' in source and 'replenishment_p95' in source
    assert 'response/unknown field supplied to selection' in source
    assert 'is_snapshot_record' in source
    manifest=json.loads((ROOT/'docs/research_pipeline/cme_orderflow_absorption_v1/oos-v1-data-manifest.json').read_text())
    assert len(manifest['chronology']['eligible_rth_dates']) == 5
    assert 'previous_rth_context' in source and 'oos_rank_recomputation_count' in source

def row(**changes):
    value={"interaction_id":"x","date":"2026-08-03","level":"PRIOR_RTH_POC","level_price":5000000000000,"end_price":5000000000000,"end_ns":1,"sell_aggressor_volume":10,"buy_aggressor_volume":5,"executions":3,"aggressive_imbalance":5,"replenishment_count":2,"replenished_volume":8,"execution_volume":10}
    value.update(changes); return value

def test_oos_scores_are_independent_and_calibration_is_deterministic():
    calibration=load_development_calibration()
    assert calibration["source"] == "DEVELOPMENT_ONLY"
    assert calibration["feature_rank_mapping"]
    assert calibration["feature_rank_mapping_encoding"] == "zlib-base64-f64le-sorted-sample"
    assert set(calibration["feature_rank_mapping"]) == {name for names in calibration["score_components"].values() for name in names}
    one=_score([row()],calibration)[0]
    changed=_score([row(),row(interaction_id="unrelated",sell_aggressor_volume=999999)],calibration)[0]
    assert (one["absorption_score"],one["replenishment_score"]) == (changed["absorption_score"],changed["replenishment_score"])
    assert _score([row()],load_development_calibration()) == _score([row()],calibration)

def test_no_oos_rank_and_missing_calibration_or_context_fail_closed(monkeypatch, tmp_path):
    import research_pipeline.cme_orderflow_absorption_v1.oos_backtest_runner as runner
    source=(ROOT/'src/research_pipeline/cme_orderflow_absorption_v1/oos_backtest_runner.py').read_text()
    assert 'def _rank(' not in source
    calibration=load_development_calibration()
    bad=dict(calibration,feature_rank_mapping={})
    path=tmp_path/'calibration.json'; path.write_text(json.dumps(bad))
    monkeypatch.setattr(runner,'CALIBRATION',path)
    try: runner.load_development_calibration()
    except RuntimeError: pass
    else: assert False
    bad=dict(calibration,previous_rth_context={})
    path.write_text(json.dumps(bad))
    try: runner.load_development_calibration()
    except RuntimeError: pass
    else: assert False

def test_prior_rth_provenance_and_observed_sessions_gate_completion():
    state=complete_state()
    state.stage="COMPLETE"
    state.counters.update({"reconstruction_stage_invoked":1,"interaction_stage_invoked":1,"scoring_stage_invoked":1,"plus_stage_invoked":1})
    state.previous_rth_source_by_oos_date={"2026-08-03":"2026-07-31","2026-08-04":"2026-08-03","2026-08-05":"2026-08-04","2026-08-06":"2026-08-05","2026-08-07":"2026-08-06"}
    assert state.previous_rth_source_by_oos_date["2026-08-03"] == "2026-07-31"
    assert state.previous_rth_source_by_oos_date["2026-08-04"] == "2026-08-03"
    calibration=load_development_calibration()
    assert set(calibration["previous_rth_context"]) == {"2026-07-31"}
    assert set(calibration["previous_rth_context"]["2026-07-31"]) == {"PRIOR_RTH_HIGH","PRIOR_RTH_LOW","PRIOR_RTH_POC","PRIOR_RTH_VAH","PRIOR_RTH_VAL"}
    # July is a read-only seed; it cannot appear in an OOS session or ledger.
    assert "2026-07-31" not in state.actual_rth_dates_processed
    assert not state.trades
    assert completed_state(state)
    state.actual_rth_dates_processed=state.actual_rth_dates_processed[:-1]
    # An expected-date union cannot manufacture a fifth observed session.
    state.counters["rth_sessions_processed"]=5
    assert not completed_state(state)
