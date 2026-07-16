# Account lifecycle

Evaluation and qualified accounts use separate ledgers. Evaluation
subscriptions, resets, failures, cancellations, and passes are billing/account
events; qualified trading profit and payout-cycle profit are not evaluation
profit. DLG locks the day, while MLL failure ends the account. The deterministic
fixture records an immediate cancellation after failure under its scenario
policy and does not keep trading after failure.

Session-close requirements are represented in the provider rule schema. The
current synthetic adapter uses fully settled intraday signals and therefore
does not claim to prove exchange-session flattening or intrabar equity behavior.
