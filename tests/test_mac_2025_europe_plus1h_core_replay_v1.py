from __future__ import annotations

import gzip
import json

import numpy as np

from research_pipeline.cme_orderflow_absorption_l2_v1 import (
    mac_2025_es_only_train_baseline as baseline,
    mac_2025_europe_plus1h_core_replay_v1 as study,
)


RAW_DTYPE = np.dtype([
    ("ts_recv", "<i8"), ("action", "S1"), ("side", "S1"),
    ("price", "<i8"), ("size", "<i4"),
    *[(f"{side}_{field}_{level:02d}", dtype)
      for level in range(10) for side in ("bid", "ask")
      for field, dtype in (("px", "<i8"), ("sz", "<i4"), ("ct", "<i4"))],
])


def _row(timestamp: int, *, bid: float, ask: float, action: bytes = b"A",
         side: bytes = b"N", price: float | None = None, size: int = 1) -> np.ndarray:
    rows = np.zeros(1, dtype=RAW_DTYPE)
    row = rows[0]
    row["ts_recv"], row["action"], row["side"] = timestamp, action, side
    row["price"] = int((price if price is not None else bid) * 1_000_000_000)
    row["size"] = size
    row["bid_px_00"], row["bid_sz_00"], row["bid_ct_00"] = int(bid * 1e9), 5, 2
    row["ask_px_00"], row["ask_sz_00"], row["ask_ct_00"] = int(ask * 1e9), 6, 2
    return row


def _shifted(day: str) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int]]]:
    base = baseline._session_windows(day)
    plus = dict(base)
    plus["EUROPE"] = (base["EUROPE"][0] + 3_600_000_000_000, base["EUROPE"][1])
    return base, plus


def test_march_7_crossed_native_snapshot_uses_policy_c_recovery() -> None:
    # Exact DBN ts_recv is 543 ns before the failure's human-readable rounded
    # timestamp; the native top levels are bid 5782.75 / ask 5778.00.
    crossed_ns = 1_741_354_203_668_377_543
    adapter = baseline.FastNativeReplayAdapter()
    assert adapter.feed_array(_row(crossed_ns - 1, bid=5782.50, ask=5782.75),
                              materialize_public=False) is None
    crossed = _row(crossed_ns, bid=5782.75, ask=5778.00)
    assert baseline._candidate_tape_in_window_book_state_supported("TEMPORARILY_NON_EXECUTABLE")
    assert adapter.feed_array(crossed, materialize_public=False) is None
    assert adapter.state == "TEMPORARILY_NON_EXECUTABLE"

    reopened = _row(crossed_ns + 5_000_000_000, bid=5776.25, ask=5776.50, side=b"B")
    public = adapter.feed_array(reopened, materialize_public=True)
    assert adapter.state == "EXECUTABLE"
    assert public is not None
    assert public.snapshot.bids[0].price == 5776.25
    assert public.snapshot.asks[0].price == 5776.50
    # The first post-crossing add is deliberately not synthesized as a delta;
    # this is the existing reopen behavior shared with the baseline adapter.
    assert public.update is None

    common = {"initial_executable_ns": crossed_ns - 20, "first_strategy_start_ns": crossed_ns - 10,
              "final_strategy_end_ns": crossed_ns + 10_000_000_000,
              "last_source_timestamp_ns": crossed_ns + 10_000_000_001,
              "sessions_seen": set(baseline.SESSION_ORDER),
              "sessions_finished": set(baseline.SESSION_ORDER)}
    # A transient interval that recovered is valid under the established
    # adapter semantics; an unresolved in-window interval remains fail-closed.
    assert baseline._candidate_tape_terminal_state_accepted(state="EXECUTABLE", **common) is False
    import pytest
    with pytest.raises(baseline.BaselineError, match="during required strategy coverage"):
        baseline._candidate_tape_terminal_state_accepted(
            state="TEMPORARILY_NON_EXECUTABLE", in_window_non_executable=True, **common
        )


def test_shifted_europe_membership_does_not_skip_raw_book_warmup_or_trades() -> None:
    for day in ("2025-03-07", "2025-10-07"):
        base, plus = _shifted(day)
        t = base["EUROPE"][0] + 30 * 60 * 1_000_000_000
        assert baseline._session_for_timestamp(t, base) == "EUROPE"
        assert baseline._session_for_timestamp(t, plus) is None

        # Even where the shifted variant has no strategy-session membership,
        # the production raw-feed helper still advances the MBP-10 adapter.
        raw = _row(t, bid=5800.00, ask=5800.25, action=b"T", side=b"B",
                   price=5800.25, size=3)
        base_adapter, plus_adapter = baseline.FastNativeReplayAdapter(), baseline.FastNativeReplayAdapter()
        base_public = baseline._feed_market_state(base_adapter, raw, session="EUROPE", action="T",
                                                  active=False, interest_prices=frozenset())
        plus_public = baseline._feed_market_state(plus_adapter, raw, session=None, action="T",
                                                  active=False, interest_prices=frozenset())
        assert base_adapter._last_timestamp_ns == plus_adapter._last_timestamp_ns == t
        assert base_adapter.state == plus_adapter.state == "EXECUTABLE"
        assert base_public is not None and plus_public is not None
        assert base_public.execution == plus_public.execution
        assert (base_public.snapshot.bids, base_public.snapshot.asks) == (
            plus_public.snapshot.bids, plus_public.snapshot.asks
        )


def test_session_window_change_changes_profile_membership_not_raw_market_input() -> None:
    day = "2025-03-07"
    base, plus = _shifted(day)
    t = base["EUROPE"][0] + 30 * 60 * 1_000_000_000
    assert baseline._session_for_timestamp(t, base) == "EUROPE"
    assert baseline._session_for_timestamp(t, plus) is None
    assert base["EUROPE"][1] == plus["EUROPE"][1]

    # Europe profiles use execution events only within the configured window.
    base_profile = baseline.Profile.create(day, "EUROPE", *base["EUROPE"])
    plus_profile = baseline.Profile.create(day, "EUROPE", *plus["EUROPE"])
    base_profile.add(5800.25, 3)
    if plus["EUROPE"][0] <= t < plus["EUROPE"][1]:
        plus_profile.add(5800.25, 3)
    assert base_profile.volume_by_tick == {5_800_250_000_000: 3}
    assert plus_profile.volume_by_tick == {}


def test_plus1h_checkpoint_identity_rejects_different_session_windows(tmp_path) -> None:
    day = "2025-03-07"
    _, plus = _shifted(day)
    path = tmp_path / "checkpoint.json.gz"
    payload = {"status": "COMPLETE", "source_sha256": "source", "variant": "PLUS1H_0900",
               "session_windows": {k: list(v) for k, v in plus.items()}, "trades": []}
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)
    assert study._read_checkpoint(path, source_sha="source", variant="PLUS1H_0900",
                                  session_windows=plus) is not None
    assert study._read_checkpoint(path, source_sha="source", variant="PLUS1H_0900",
                                  session_windows=baseline._session_windows(day)) is None


def test_plus1h_profile_checkpoint_identity_uses_embedded_profile_windows(tmp_path) -> None:
    day = "2025-03-07"
    _, plus = _shifted(day)
    path = tmp_path / "profile.json.gz"
    payload = {"status": "COMPLETE", "source_sha256": "source", "variant": "PLUS1H_PROFILE",
               "profiles": [{"session": session, "start_ns": start, "end_ns": end}
                            for session, (start, end) in plus.items()]}
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)
    assert study._read_checkpoint(path, source_sha="source", variant="PLUS1H_PROFILE",
                                  session_windows=plus) is not None
    assert study._read_checkpoint(path, source_sha="source", variant="PLUS1H_PROFILE",
                                  session_windows=baseline._session_windows(day)) is None
