# Holdout execution

The holdout can be opened only in `HOLDOUT`. The Phase A durable lock records
reason, timestamp, dataset hash, and access count. The default maximum is one.
After opening, parameters are frozen and parameter research cannot resume.
Holdout results never drive parameter reselection; they only provide final
validation evidence.
