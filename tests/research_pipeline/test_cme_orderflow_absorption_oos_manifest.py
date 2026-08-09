import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "docs/research_pipeline/cme_orderflow_absorption_v1/oos-v1-data-manifest.json"


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_oos_manifest_is_json_and_seals_only_the_first_post_pilot_week():
    manifest = load_manifest()
    chronology = manifest["chronology"]
    assert chronology["oos_signal_and_performance_interval"] == {
        "start": "2026-08-03T00:00:00Z",
        "end": "2026-08-08T00:00:00Z",
    }
    assert chronology["eligible_rth_dates"] == [
        "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"
    ]
    assert len(chronology["eligible_rth_dates"]) == 5
    assert chronology["pilot"]["role"] == "DEVELOPMENT_DESIGN_ONLY"
    assert chronology["july_31_warmup"]["oos_excluded"] is True


def test_oos_manifest_seals_owner_obtained_databento_resolution_and_cost_without_acquisition():
    manifest = load_manifest()
    assert manifest["status"] == "COST_QUOTED_NOT_ACQUIRED"
    assert manifest["provider"]["dataset"] == "GLBX.MDP3"
    assert manifest["provider"]["schema"] == "mbo"
    assert manifest["provider"]["stype_in"] == "raw_symbol"
    evidence = manifest["provider"]["metadata_symbology_evidence"]
    assert evidence["provenance"] == "Owner-obtained official Databento evidence"
    assert evidence["result"] == {"status": "OK", "raw_symbol": "ESU6", "instrument_id": 42140870, "partial": [], "not_found": []}
    assert manifest["symbol_and_instrument"]["resolved_raw_symbol"] == "ESU6"
    assert manifest["symbol_and_instrument"]["instrument_ids"] == [42140870]
    assert manifest["rollover"]["resolved"] is True
    assert manifest["rollover"]["contracts"] == ["ESU6"]
    assert manifest["authorized_cost_quote"]["quote_count"] == 1
    assert manifest["authorized_cost_quote"]["estimated_usd"] == 5.736491556466
    assert manifest["proposed_acquisition"]["target_path"] == "data/cme_orderflow_absorption_v1/oos_v1/ESU6/mbo/ESU6_2026-08-03_2026-08-08_mbo.dbn"
    assert manifest["proposed_acquisition"]["no_file_created"] is True
    assert manifest["data_acquired"] is False
    assert manifest["no_strategy_execution_or_pnl"] is True
    assert manifest["fixed_usd_risk_budget"] == 250.00
    assert manifest["stop_buffer_ticks"] == 5


def test_oos_manifest_preserves_frozen_plus_selection_without_refit():
    manifest = load_manifest()
    frozen = manifest["frozen_strategy"]
    assert frozen["absorption_p95"] == 0.7977986403366786
    assert frozen["replenishment_p95"] == 0.7785691162188411
    assert "no refit" in frozen["constraints"][0]
    assert "No strategy execution, backtest, PnL calculation" in frozen["constraints"][1]
