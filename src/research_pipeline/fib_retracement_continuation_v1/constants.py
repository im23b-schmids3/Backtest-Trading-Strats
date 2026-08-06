from decimal import Decimal

STRATEGY_ID = "FibRetracementContinuation.ETH_BTC_V1_PROSPECTIVE_VALIDATION"
SPEC_PATH = ".smithers/specs/fib-retracement-continuation-eth-btc-v1-prospective-validation.md"
ENTRY_RATIO = Decimal(".900")
STOP_RATIO = Decimal("1.020")
TARGET_RATIOS = (Decimal(".786"), Decimal(".618"), Decimal(".500"), Decimal(".236"), Decimal(".050"))
TARGET_FRACTIONS = (Decimal(".30"), Decimal(".25"), Decimal(".20"), Decimal(".15"), Decimal(".10"))
TERMINAL_OUTCOMES = frozenset(("TRADE_EXECUTED", "DIRECTION_REJECTED", "IMPULSE_NOT_CONFIRMED", "FIB_RANGE_INVALID", "ENTRY_NOT_REACHED", "ENTRY_EXPIRED", "STOP_DISTANCE_REJECTED", "ACTIVE_POSITION_BLOCKED", "SESSION_OR_DATA_END", "DATA_CONTRACT_BLOCKED"))
EVIDENCE_LABELS = ((0, 29, "LOW_FREQUENCY_DEVELOPMENT_EVIDENCE"), (30, 59, "MODERATE_DEVELOPMENT_EVIDENCE"), (60, None, "FULL_DEVELOPMENT_GATE_ELIGIBILITY"))
CANDIDATES = (
 {"candidate_id":"FIB09-ETH-4H-POST0830","symbol":"ETH","timeframe":"4h","post_tp1_ratio":Decimal(".830"),"min_distance":16,"min_move":Decimal(".0025"),"anchor_age_days":60},
 {"candidate_id":"FIB09-ETH-4H-POST0786-REFERENCE","symbol":"ETH","timeframe":"4h","post_tp1_ratio":Decimal(".786"),"min_distance":16,"min_move":Decimal(".0025"),"anchor_age_days":60},
 {"candidate_id":"FIB09-BTC-1D-POST0786","symbol":"BTC","timeframe":"1d","post_tp1_ratio":Decimal(".786"),"min_distance":7,"min_move":Decimal(".0025"),"anchor_age_days":180},
)
