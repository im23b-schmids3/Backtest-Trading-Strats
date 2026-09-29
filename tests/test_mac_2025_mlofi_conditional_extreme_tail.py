import numpy as np

from research_pipeline.cme_orderflow_absorption_l2_v1.mac_2025_mlofi_conditional_extreme_tail import (
    GROUPS,
    RangeRMQ,
    TailAccumulator,
    _bucket,
    _classes,
    _group_mask,
)


def test_tail_bucket_partition_is_deterministic_and_left_closed():
    cuts = [1.0, 2.0, 3.0]
    values = np.asarray([0.0, 1.0, 1.0001, 2.0, 3.0, 4.0])
    assert _bucket(values, cuts).tolist() == [0, 0, 1, 1, 2, 3]


def test_directional_tail_accumulator_flips_negative_signal_outcomes():
    accumulator = TailAccumulator(groups=1, views=1)
    accumulator.update(0, 0, 0, np.asarray([0, 0]), np.asarray([2.0, -3.0])[:, None], np.asarray([1, 2]))
    assert accumulator.exact[0, 0, 0, 0, 0, 0] == 2
    assert accumulator.exact[0, 0, 0, 0, 0, 1] == -1


def test_range_rmq_matches_naive_values_and_absolute_arg_indices():
    values = np.asarray([5, 2, 9, 1, 7, 3, 8, 0, 6, 4, 10, 1, 11], dtype=np.int32)
    rmq = RangeRMQ(values, block_size=4)
    left = np.asarray([0, 1, 2, 3, 5, 0, 7, 10])
    right = np.asarray([3, 5, 7, 8, 11, 13, 13, 13])
    maximum, maximum_index, minimum, minimum_index = rmq.query(left, right)
    for row, (lo, hi) in enumerate(zip(left, right)):
        segment = values[lo:hi]
        assert maximum[row] == segment.max()
        assert minimum[row] == segment.min()
        assert maximum_index[row] == lo + int(segment.argmax())
        assert minimum_index[row] == lo + int(segment.argmin())


def test_range_rmq_handles_empty_ranges_fail_closed():
    rmq = RangeRMQ(np.asarray([1, 2, 3], dtype=np.int32), block_size=2)
    maximum, maximum_index, minimum, minimum_index = rmq.query(np.asarray([1]), np.asarray([1]))
    assert maximum_index.tolist() == [-1]
    assert minimum_index.tolist() == [-1]
    assert maximum[0] < 0
    assert minimum[0] > 0


def test_regime_groups_keep_europe_and_ny_semantics_separate():
    session = np.asarray([1, 1, 1, 2, 0])
    depth = np.asarray([0, 2, 1, 0, 0])
    volatility = np.asarray([0, 0, 2, 0, 0])
    assert _group_mask(0, session, depth, volatility).tolist() == [True, False, False, False, False]
    assert _group_mask(1, session, depth, volatility).tolist() == [False, True, False, False, False]
    assert _group_mask(2, session, depth, volatility).tolist() == [False, False, True, False, False]
    assert _group_mask(3, session, depth, volatility).tolist() == [False, False, False, True, False]
    assert _group_mask(4, session, depth, volatility).tolist() == [True, True, True, True, True]
    assert len(GROUPS) == 5


def test_class_boundaries_are_low_medium_high_from_q10_to_q90():
    calibration = {
        "depth_percentile_cuts_q10_to_q90": list(range(1, 10)),
        "volatility_percentile_cuts_q10_to_q90": list(range(1, 10)),
    }
    depth_class, volatility_class = _classes(np.asarray([0.0, 5.0, 9.0, 20.0]),
                                             np.asarray([0.0, 5.0, 9.0, 20.0]), calibration)
    assert depth_class.tolist() == [0, 1, 2, 2]
    assert volatility_class.tolist() == [0, 1, 2, 2]
