from __future__ import annotations

import pytest

from research_pipeline.cme_orderflow_absorption_l2_v1 import mac_2025_es_only_train_baseline as baseline
from research_pipeline.cme_orderflow_absorption_l2_v1 import ten_family_dec_jan_robustness as robustness


WINDOWS = set(baseline.SESSION_ORDER)


def _validate(**overrides):
    values = {
        "state": "TEMPORARILY_NON_EXECUTABLE",
        "initial_executable_ns": 0,
        "first_strategy_start_ns": 0,
        "final_strategy_end_ns": 100,
        "last_source_timestamp_ns": 101,
        "sessions_seen": set(WINDOWS),
        "sessions_finished": set(WINDOWS),
    }
    values.update(overrides)
    return baseline._candidate_tape_terminal_state_accepted(**values)


def test_post_session_locked_crossed_terminal_state_is_candidate_tape_only_exception():
    assert _validate() is True


@pytest.mark.parametrize("session", ["ASIA", "EUROPE", "NY"])
def test_non_executable_state_inside_each_strategy_session_remains_fatal(session):
    del session  # The route applies this invariant uniformly to every active window.
    with pytest.raises(baseline.BaselineError, match="during required strategy coverage"):
        _validate(in_window_non_executable=True)


def test_non_executable_state_at_or_before_final_window_end_is_fatal():
    with pytest.raises(baseline.BaselineError, match="during required strategy coverage"):
        _validate(last_source_timestamp_ns=100, in_window_non_executable=True)


def test_non_executable_state_strictly_after_final_window_end_is_accepted():
    assert _validate(last_source_timestamp_ns=101) is True


def test_source_truncated_before_final_required_window_end_is_fatal():
    with pytest.raises(baseline.BaselineError, match="ended before the final strategy window"):
        _validate(last_source_timestamp_ns=99, state="EXECUTABLE")


def test_parser_or_integrity_failure_before_cutoff_is_fatal():
    with pytest.raises(baseline.BaselineError, match="integrity failure"):
        _validate(state="EXECUTABLE", integrity_error="synthetic parser failure")


def test_missing_required_session_is_fatal_even_if_terminal_state_is_post_session():
    with pytest.raises(baseline.BaselineError, match="incomplete required strategy windows"):
        _validate(sessions_finished={"ASIA", "EUROPE"})


def test_initial_book_must_be_executable_before_strategy_starts():
    with pytest.raises(baseline.BaselineError, match="not executable before strategy processing"):
        _validate(initial_executable_ns=1)


def test_shared_adapter_finish_remains_strict_for_non_executable_eof():
    adapter = baseline.FastNativeReplayAdapter()
    adapter.first_valid_book_ns = 0
    adapter.state = "TEMPORARILY_NON_EXECUTABLE"
    with pytest.raises(baseline.BaselineError, match="incomplete native MBP-10 source"):
        adapter.finish()


def test_strict_reference_event_comparison_uses_candidate_tape_sparse_contract():
    rows = [
        {"timestamp_ns": 1, "bid": 100.0, "ask": 100.25, "execution_price": None,
         "execution_size": 0, "aggressor": None, "session": "ASIA"},
        # Unchanged BBO, no execution: correctly omitted by the candidate-tape spool.
        {"timestamp_ns": 2, "bid": 100.0, "ask": 100.25, "execution_price": None,
         "execution_size": 0, "aggressor": None, "session": "ASIA"},
        {"timestamp_ns": 3, "bid": 100.0, "ask": 100.25, "execution_price": 100.25,
         "execution_size": 1, "aggressor": "BUY", "session": "ASIA"},
    ]
    compact = robustness._compact_reference_events(rows)
    assert len(compact) == 2
    assert robustness._event_arrays_equal(compact, compact.copy())


def test_strict_reference_event_comparison_treats_corresponding_nan_as_equal():
    import numpy as np

    dtype = [("timestamp_ns", "<i8"), ("execution_price", "<f8")]
    left = np.array([(1, np.nan)], dtype=dtype)
    right = np.array([(1, np.nan)], dtype=dtype)
    assert robustness._event_arrays_equal(left, right)
