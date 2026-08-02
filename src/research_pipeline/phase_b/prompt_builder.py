from __future__ import annotations

import json

from ..schemas.strategy_spec import StrategySpec


def build_spec_agent_prompt(strategy_name: str, description: str, markets: list[str], timeframes: list[str], notes: str | None) -> str:
    return f"""Create only one brand-new structured YAML or JSON strategy specification for Phase A.
The exact required strategy_id is {strategy_name!r}; the exact required name is {strategy_name!r}.
Do not reuse, copy, or reference any existing strategy specification or fixture.
Strategy name: {strategy_name}
Natural-language description: {description}
Requested markets: {', '.join(markets)}
Requested timeframes: {', '.join(timeframes)}
Notes: {notes or '(none)'}

Return exactly one mapping compatible with research_pipeline.schemas.strategy_spec.StrategySpec.
The response must contain no prose, no multiple candidates, no competing values,
and no commentary before or after the mapping. Markdown fences are unnecessary.
Do not call tools, inspect files, create files, apply patches, or modify the
workspace. This is a response-only serialization task: construct the single
mapping directly in your final response.
Use explicit provenance in session_assumptions/known_limitations when translating
technical details; distinguish confirmed rules from technical translations and
assumptions. Treat unresolved material ambiguity as blocking and do not guess.
Do not output Fibonacci, compatibility, or placeholder strategy content. Do not implement code, run backtests, optimize parameters, or infer material
rules that are absent from the description. Put unresolved material uncertainty
in known_limitations and identify ambiguities in the metadata. The YAML must
contain all required StrategySpec fields, status DRAFT, approved_at null, and a
canonical specification_hash calculated after normalization.
Set specification_hash to the literal string "pending"; the deterministic
Python validator calculates and verifies the canonical hash. Do not calculate
or guess the SHA-256 value yourself.
If SPY is requested, explicitly state that it is the repository proxy and that
existing Phase D futures mappings do not support SPY; do not invent a mapping.
For the named RandomOpenTest integration strategy, use strategy_family exactly
"f2_random_open_test" so the existing repository-compatible adapter can be
resolved. Do not invent another family name.
For RandomOpenTest, baseline_parameters must include equity_fraction: 0.05,
initial_cash: 10000, session_timezone: America/New_York, session_open_local:
"09:30", test_start_date: "2025-01-01", and test_end_date: "2026-01-01".
Describe the 5% rule as allocation from current account equity, never as risk
per trade. Preserve the one-hour same-bar exit and explicit proxy disclosure.
"""


def build_implementation_prompt(specification: StrategySpec, plan_files: list[str], required_tests: list[list[str]], budget: int) -> str:
    material = specification.model_dump(mode="json")
    material.pop("approved_at", None)
    material.pop("specification_hash", None)
    material.pop("status", None)
    return f"""Implement the approved strategy specification in an isolated worktree.
Current pipeline phase: IMPLEMENTATION
Approved specification (material fields only):
{json.dumps(material, indent=2, sort_keys=True)}

Allowed file areas: {plan_files}
Required technical tests: {required_tests}
Repair budget: {budget}
Invariants (must remain unchanged):
{chr(10).join('- ' + item for item in specification.invariants)}

For real-mode work, create or register the canonical structured strategy
adapter, expose trade-level diagnostics, and write implementation metadata with
files created/modified, adapter registration, strategy entry point, tests
added, verification command, known limitations, unresolved ambiguities, and
the worktree commit hash. Use only the bounded parameter families declared in
the approved specification.

Do not run backtests or optimization during implementation. Do not inspect or use holdout results.
Do not modify existing Fibonacci strategy logic, research versions V1-V13,
market-data providers, futures mappings, Alpha Futures lifecycle logic, or
approved invariants. Add only the new strategy implementation and focused
tests/adapters required by the approved specification. If the specification is
materially ambiguous or conflicts with an invariant, stop and report it.
"""
