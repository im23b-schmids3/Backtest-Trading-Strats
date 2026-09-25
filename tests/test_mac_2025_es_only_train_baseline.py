from datetime import date
from types import SimpleNamespace

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_es_only_train_baseline as baseline
from research_pipeline.cme_orderflow_absorption_l2_v1.public_book_adapters import NativeDatabentoMBP10Adapter


def test_session_windows_use_new_york_dst_without_lookahead() -> None:
    march = baseline._session_windows("2025-03-07")
    after_dst = baseline._session_windows("2025-03-10")
    assert march["EUROPE"][1] - march["EUROPE"][0] == 6 * 60 * 60 * 1_000_000_000 + 30 * 60 * 1_000_000_000
    assert after_dst["EUROPE"][1] - after_dst["EUROPE"][0] == 5 * 60 * 60 * 1_000_000_000 + 30 * 60 * 1_000_000_000
    assert march["NY"][0] < march["NY"][1] and after_dst["NY"][0] < after_dst["NY"][1]


def test_profile_is_executed_volume_profile_with_lower_poc_tie() -> None:
    profile = baseline.Profile.create("2025-03-03", "ASIA", 0, 1)
    profile.add(5000.25, 10)
    profile.add(5000.00, 10)
    profile.add(5000.50, 5)
    values = profile.values()
    assert values["POC"] == 5000.00
    assert values["LOW"] == 5000.00
    assert values["HIGH"] == 5000.50


def test_family_matrix_contains_required_cross_session_and_dynamic_cells() -> None:
    prior = {session: baseline.Profile.create("2025-02-28", session, 0, 1) for session in baseline.SESSION_ORDER}
    current = {session: baseline.Profile.create("2025-03-03", session, 0, 1) for session in baseline.SESSION_ORDER}
    families = baseline.build_families("2025-03-03", prior, current)
    keys = {(item.trading_session, item.reference_session, item.reference_day, item.reference_level) for item in families}
    assert ("EUROPE", "ASIA", "CURRENT", "HIGH") in keys
    assert ("NY", "EUROPE", "CURRENT", "VAH") in keys
    assert ("NY", "RTH", "PRIOR", "POC") not in keys
    assert sum(item.dynamic for item in families) == 6
    assert all(item.causal_ready_timestamp.endswith("Z") for item in families)


def test_train_plan_is_exactly_train_only() -> None:
    assert len(baseline.TRAIN_DATES) == 35
    assert baseline.DEPENDENCY_DATE == "2025-02-28"
    assert "2025-10-07" not in baseline.TRAIN_DATES
    assert date.fromisoformat(baseline.TRAIN_DATES[0]) == date(2025, 3, 3)


def test_fast_replay_adapter_matches_native_public_trade_semantics() -> None:
    def level(bid_size: int = 10, ask_size: int = 10) -> SimpleNamespace:
        return SimpleNamespace(
            bid_px=5_000_000_000, bid_sz=bid_size, bid_ct=2,
            ask_px=5_000_250_000, ask_sz=ask_size, ask_ct=2,
        )

    add = SimpleNamespace(ts_recv=1, action="A", side="B", price=5_000_000_000, size=10, levels=(level(),))
    trade = SimpleNamespace(ts_recv=2, action="T", side="B", price=5_000_250_000, size=3, levels=(level(10, 7),))
    native = NativeDatabentoMBP10Adapter()
    fast = baseline.FastNativeReplayAdapter()
    native.feed(add)
    fast.feed(add, materialize_public=True)
    expected = native.feed(trade)
    actual = fast.feed(trade, materialize_public=True)
    assert expected is not None and actual is not None
    assert actual.execution == expected.execution
    assert [(row.price, row.size, row.order_count) for row in actual.snapshot.bids] == [
        (row.price, row.size, row.order_count) for row in expected.snapshot.bids
    ]
    assert [(row.price, row.size, row.order_count) for row in actual.snapshot.asks] == [
        (row.price, row.size, row.order_count) for row in expected.snapshot.asks
    ]
    assert actual.update is not None and expected.update is not None
    assert (actual.update.side, actual.update.price, actual.update.size_delta, actual.update.kind) == (
        expected.update.side, expected.update.price, expected.update.size_delta, expected.update.kind,
    )


def test_profile_cache_round_trip_preserves_executed_volume(tmp_path) -> None:
    profile = baseline.Profile.create("2025-03-03", "ASIA", 1, 2)
    profile.add(5000.00, 10)
    profile.add(5000.25, 3)
    path = tmp_path / "profile.json"
    baseline._write_profile_cache(path, {"ASIA": profile}, day="2025-03-03",
                                  source_path=tmp_path / "source.dbn.zst", source_sha256="source",
                                  semantic_sha256="semantic")
    loaded = baseline._load_profile_cache(path, day="2025-03-03", source_path=tmp_path / "source.dbn.zst",
                                          source_sha256="source", semantic_sha256="semantic")
    assert loaded is not None
    assert loaded["ASIA"].volume_by_tick == profile.volume_by_tick


def test_checkpoint_rejects_source_or_semantic_hash_change(tmp_path) -> None:
    profile = baseline.Profile.create("2025-03-03", "ASIA", 1, 2)
    profile.add(5000.00, 10)
    result = {"date": "2025-03-03", "records": 1, "profiles": {"ASIA": profile}, "families": (),
              "interactions": [], "setups": [], "trades": [], "session_metrics": [],
              "family_counts": {}, "family_trades": {}}
    path = tmp_path / "checkpoint.json"
    source = tmp_path / "source.dbn.zst"
    baseline._json_write(path, baseline._checkpoint_payload(result, source_path=source, source_sha256="source",
                                                             config_sha256="config", semantic_sha256="semantic"))
    assert baseline._load_valid_checkpoint(path, day="2025-03-03", source_path=source, source_sha256="source",
                                           config_sha256="config", semantic_sha256="semantic") is not None
    assert baseline._load_valid_checkpoint(path, day="2025-03-03", source_path=source, source_sha256="changed",
                                           config_sha256="config", semantic_sha256="semantic") is None
