from ..schemas.budgets import ResearchBudget
from ..schemas.gates import Comparison, GateDefinition, GateSet

DEFAULT_BUDGETS = ResearchBudget()
DEFAULT_GATES = GateSet(gates=[
    GateDefinition(name="baseline_minimum_trades", category="baseline", metric="completed_trades", threshold=30, comparison=Comparison.GREATER_EQUAL),
    GateDefinition(name="baseline_profit_factor", category="baseline", metric="profit_factor", threshold=1.1, comparison=Comparison.GREATER_EQUAL),
    GateDefinition(name="baseline_expectancy_r", category="baseline", metric="expectancy_r", threshold=0, comparison=Comparison.GREATER_EQUAL),
    GateDefinition(name="baseline_fee_share", category="baseline", metric="fee_share_of_gross_profit", threshold=0.5, comparison=Comparison.LESS_EQUAL),
    GateDefinition(name="baseline_drawdown", category="baseline", metric="max_drawdown", threshold=0.25, comparison=Comparison.LESS_EQUAL),
    GateDefinition(name="validation_profitable_fold_ratio", category="validation", metric="profitable_fold_ratio", threshold=0.5, comparison=Comparison.GREATER_EQUAL),
    GateDefinition(name="validation_minimum_trades", category="validation", metric="validation_trades", threshold=20, comparison=Comparison.GREATER_EQUAL),
    GateDefinition(name="validation_drawdown", category="validation", metric="validation_drawdown", threshold=0.25, comparison=Comparison.LESS_EQUAL),
    GateDefinition(name="validation_profit_factor", category="validation", metric="validation_profit_factor", threshold=1.0, comparison=Comparison.GREATER_EQUAL),
    GateDefinition(name="holdout_minimum_trades", category="holdout", metric="holdout_trades", threshold=20, comparison=Comparison.GREATER_EQUAL),
    GateDefinition(name="holdout_expectancy_r", category="holdout", metric="holdout_expectancy_r", threshold=0, comparison=Comparison.GREATER_EQUAL),
    GateDefinition(name="holdout_drawdown", category="holdout", metric="holdout_drawdown", threshold=0.25, comparison=Comparison.LESS_EQUAL),
    GateDefinition(name="holdout_profit_factor", category="holdout", metric="holdout_profit_factor", threshold=1.0, comparison=Comparison.GREATER_EQUAL),
    GateDefinition(name="throughput_trades_per_month", category="throughput", metric="executable_trades_per_month", threshold=5, comparison=Comparison.GREATER_EQUAL),
    GateDefinition(name="throughput_median_days", category="throughput", metric="median_days_between_trades", threshold=31, comparison=Comparison.LESS_EQUAL),
    GateDefinition(name="throughput_zero_trade_months", category="throughput", metric="zero_trade_month_percentage", threshold=0.5, comparison=Comparison.LESS_EQUAL),
])

