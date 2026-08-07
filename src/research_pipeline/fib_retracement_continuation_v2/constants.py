from decimal import Decimal
STRATEGY_ID="FibRetracementContinuation.ETH_BTC_V2_INTRADAY_FORCE_FLAT_2245"
NO_HOLDOUT_LOGICAL_EXPOSURE="NO_HOLDOUT_LOGICAL_EXPOSURE"
DEVELOPMENT_START="2022-01-01T00:00:00+00:00"; DEVELOPMENT_END="2025-01-01T00:00:00+00:00"
CANDIDATES=({"candidate_id":"FIB09-V2-ETH-4H-POST0830","symbol":"ETH","timeframe":"4h","post_tp1_ratio":Decimal('.830'),"min_distance":16,"min_move":Decimal('.0025'),"anchor_age_days":60},{"candidate_id":"FIB09-V2-ETH-4H-POST0786-REFERENCE","symbol":"ETH","timeframe":"4h","post_tp1_ratio":Decimal('.786'),"min_distance":16,"min_move":Decimal('.0025'),"anchor_age_days":60},{"candidate_id":"FIB09-V2-BTC-1D-POST0786","symbol":"BTC","timeframe":"1d","post_tp1_ratio":Decimal('.786'),"min_distance":7,"min_move":Decimal('.0025'),"anchor_age_days":180})
