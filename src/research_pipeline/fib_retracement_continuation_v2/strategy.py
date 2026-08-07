"""V2 deliberately has no signal implementation of its own.

Signals, identities, prices and expiry semantics are frozen V1 code.  V2 only
adapts completed 1m-derived higher-timeframe bars to that public interface.
"""
from research_pipeline.fib_retracement_continuation_v1.strategy import (  # noqa: F401
    causal_setups, create_setup, d, expire_reason, fib_price, touch,
)
