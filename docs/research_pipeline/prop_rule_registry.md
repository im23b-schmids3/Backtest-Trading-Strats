# Prop rule registry

`PropRuleSet` is the provider-independent contract for rule facts. The current
registry contains the verified Alpha Futures Zero 25K and Zero 50K products;
there is no default 100K product. Rules include source URLs, verification date,
source hash, account/MLL/DLG/target values, contract limits, payout rules,
fees, billing, cancellation, session assumptions, and unresolved ambiguities.

Use `python -m research_pipeline prop verify-rules STRATEGY_ID --product "Alpha Futures Zero 25K"`.
