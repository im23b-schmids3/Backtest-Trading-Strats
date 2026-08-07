"""V2 may add session mechanics, never strategy economics."""
from datetime import time
from research_pipeline.fib_retracement_continuation_v1.constants import (  # noqa: F401
    CANDIDATES, ENTRY_RATIO, EVIDENCE_LABELS, STOP_RATIO, STRATEGY_ID,
    TARGET_FRACTIONS, TARGET_RATIOS,
)

V2_STRATEGY_ID = "FibRetracementContinuation.ETH_BTC_V2_INTRADAY_FORCE_FLAT_2245"
NO_HOLDOUT_LOGICAL_EXPOSURE = "NO_HOLDOUT_LOGICAL_EXPOSURE"
DEVELOPMENT_START = "2022-01-01T00:00:00+00:00"
DEVELOPMENT_END = "2025-01-01T00:00:00+00:00"
SESSION_CUTOFF = time(22, 45)
