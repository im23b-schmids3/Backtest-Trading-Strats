# Intake schema

Intake YAML supports `strategy_name`, `description`, `markets`, `timeframes`,
`entry_logic`, `exit_logic`, `risk_model`, `position_sizing`, `filters`,
`optional_notes`, `unknown_fields`, `confidence_flags`, `ambiguities`,
`confirmed_facts`, `assumptions`, and `missing_information`.

Confirmed facts, assumptions, and missing information are kept separate. Any
material ambiguity or missing information produces `MANUAL_REVIEW_REQUIRED`
and cannot reach approval. The specification agent is instructed to preserve
unknowns rather than invent rules.
