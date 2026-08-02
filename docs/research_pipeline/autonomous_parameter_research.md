# Bounded parameter research

Research is sequential. The analyst can select one mutable parameter family
at a time, and every proposed value is checked against the specification and
budget before the adapter runs. Invariants and immutable families are never
eligible. Each family has bounded rounds and each round has at most five
values.

The deterministic fixture uses typed analyst and statistical-review records.
Production model roles may provide proposals, but the Python controller
validates them and remains authoritative.
