# Deployment classification

Phase D classifications are:

- `PROP_ACCEPTED_STANDALONE`: Phase C standalone evidence plus compliant,
  positive prop economics and at least one payout in the scenario.
- `PROP_ACCEPTED_PORTFOLIO_COMPONENT`: the same evidence when Phase C marked
  the candidate as a component.
- `OWN_CAPITAL_ONLY`: trading evidence exists but external prop economics are
  negative.
- `REJECTED_PROP_INCOMPATIBLE` or `REJECTED_NEGATIVE_ECONOMICS`: policy or
  economics fail.
- `INSUFFICIENT_FUTURES_DATA` or `INSUFFICIENT_PROP_EVIDENCE`: lifecycle or
  native-data evidence is incomplete.

Proxy and synthetic runs remain exploratory/insufficient even when their
arithmetic produces a payout. A real deployment classification requires native
futures data, operational review, and current provider terms.
