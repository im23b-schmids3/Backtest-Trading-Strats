# Intake ambiguities

Facts supplied by the user, technical translations, assumptions, missing information, and unresolved ambiguities are stored separately in `SpecificationProvenance`. Technical translations may explain repository conventions—for example, mapping a local 09:30 America/New_York cash open to timestamp handling—but may not change the trading rule.

Words such as “maybe”, “unclear”, “ambiguous”, and “unspecified”, plus explicit `missing_information` or `ambiguities` fields, produce a blocking ambiguity. The generated draft and its evidence remain auditable, but `manual_review_required` is true and approval is unavailable. The operator must clarify the intake and create a new deterministic run rather than allowing the agent to guess.

Use the inspection commands below for a run:

```text
python -m research_pipeline specification status <RUN_ID>
python -m research_pipeline specification attempts <RUN_ID>
python -m research_pipeline specification errors <RUN_ID>
python -m research_pipeline specification latest <RUN_ID>
```
