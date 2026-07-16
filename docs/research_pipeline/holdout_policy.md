# Holdout policy

A split is created before parameter research begins. Its source-data hash and
chronological training, validation, and untouched-holdout windows are hashed
and stored durably. Re-registering a different split for the same strategy
version raises a conflict.

The holdout lock can open only while the strategy is in `HOLDOUT`. The default
budget permits exactly one access, records its timestamp, phase, dataset hash,
and reason, and freezes all mutable parameter families. This is process
enforcement and auditability, not cryptographic secrecy from repository
owners. A changed specification must use a new strategy version and therefore
a new lock and split.

