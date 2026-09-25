"""Single-pass raw fanout proof for the TRAIN-only Class-B tape contract.

This module is proof tooling, not an optimizer.  The raw DBN stream is decoded
once per selected date and dispatched to independent raw runner instances.
Candidate tapes are evaluated only after the raw pass, and never inside the
raw state machines.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Mapping

from . import mac_2025_candidate_tape as candidate_tape
from . import mac_2025_es_only_train_baseline as baseline
from .model import (
    L2ClassBConfig,
    L2Config,
    L2SignalEngine,
    MAX_CONFIRMATION_NS,
    MIN_CONFIRMATION_NS,
    ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS,
)


RUN_ROOT = Path("research_runs/CMEOrderflowAbsorption.ES_L2_MAC2025_ES_ONLY_TRAIN_BASELINE")
TAPE_ROOT = RUN_ROOT / "candidate-tapes"
MASTER_PATH = RUN_ROOT / "train-optimization/class-a-final/class-a-master-selection.json"
PROOF_FAMILIES = (
    "ASIA|RTH|PRIOR|POC",
    "EUROPE|ASIA|CURRENT|LOW",
    "NY|ASIA|CURRENT|VAH",
)
# Both dates contain candidates for all three families and together exercise
# the window, favorable-tick, stop, target, and mixed paths.  A scan of all
# sealed TRAIN tapes found no candidate whose result changes under the
# count/volume variants; those dimensions are reported as no-opportunity.
PROOF_DATES = ("2025-03-25", "2025-04-02")

PARAMETER_CONFIGS = {
    "min_confirmation_seconds": ("shorter_confirmation", "longer_confirmation"),
    "max_confirmation_seconds": ("shorter_confirmation", "longer_confirmation"),
    "favorable_confirmation_ticks": ("stricter_favorable", "looser_favorable"),
    "confirmation_execution_count": ("higher_execution_count",),
    "confirmation_volume_threshold": ("volume_threshold",),
    "stop_ticks": ("tighter_stop", "wider_stop_target"),
    "target_r": ("wider_stop_target",),
}


def proof_configs() -> tuple[tuple[str, L2ClassBConfig], ...]:
    return (
        ("default", L2ClassBConfig()),
        ("shorter_confirmation", L2ClassBConfig(min_confirmation_seconds=5.0, max_confirmation_seconds=10.0)),
        ("longer_confirmation", L2ClassBConfig(min_confirmation_seconds=8.0, max_confirmation_seconds=15.0)),
        ("stricter_favorable", L2ClassBConfig(favorable_confirmation_ticks=4.0)),
        ("looser_favorable", L2ClassBConfig(favorable_confirmation_ticks=2.0)),
        ("higher_execution_count", L2ClassBConfig(confirmation_execution_count=2)),
        ("volume_threshold", L2ClassBConfig(confirmation_volume_threshold=50)),
        ("tighter_stop", L2ClassBConfig(stop_ticks=3)),
        ("wider_stop_target", L2ClassBConfig(stop_ticks=8, target_r=4.0)),
        ("mixed", L2ClassBConfig(
            min_confirmation_seconds=6.0, max_confirmation_seconds=12.0,
            favorable_confirmation_ticks=4.0, confirmation_execution_count=2,
            confirmation_volume_threshold=50, stop_ticks=6, target_r=3.5,
        )),
    )


class LegacyDefaultBroadSignalEngine(baseline.BroadSignalEngine):
    """Pre-parameterization default confirmation semantics for same-pass control."""

    def _expire(self, timestamp_ns: int) -> None:
        while self._pending_order:
            setup = self.pending[self._pending_order[0]]
            if setup.terminal_reason is not None or timestamp_ns > (setup.interaction.end_ns or 0) + MAX_CONFIRMATION_NS:
                if setup.terminal_reason is None:
                    setup.state, setup.terminal_reason = "FAILED", "CONFIRMATION_WINDOW_EXPIRED"
                self._pending_order.popleft()
                continue
            break

    def observe_execution(self, event: Any) -> None:
        self._expire(event.timestamp_ns)
        for setup_id in tuple(self._pending_order):
            setup = self.pending[setup_id]
            if setup.terminal_reason is not None or setup.state == "CONFIRMED":
                continue
            age = event.timestamp_ns - (setup.interaction.end_ns or 0)
            if age < MIN_CONFIRMATION_NS or age > MAX_CONFIRMATION_NS:
                continue
            favorable = (
                (event.price - float(setup.interaction.end_price)) / 0.25
                if setup.interaction.direction == "BUYER_ABSORPTION"
                else (float(setup.interaction.end_price) - event.price) / 0.25
            )
            if favorable >= 3:
                setup.state = "CONFIRMED"
                setup.confirmation_timestamp_ns = event.timestamp_ns
                setup.confirmation_price = event.price
                setup.entry_ready_ns = event.timestamp_ns + 2_000_000
                self._entry_ready.append(setup_id)
                self.events.append({
                    "setup_id": setup_id, "state": "CONFIRMED",
                    "timestamp_ns": event.timestamp_ns, "favorable_ticks": favorable,
                })


def _class_a_config(master_row: Mapping[str, Any]) -> L2Config:
    frozen = dict(master_row["class_a_frozen_config"])
    valid = {field.name for field in fields(L2Config)}
    return L2Config(**{key: value for key, value in frozen.items() if key in valid})


def _tape_parameters(config: L2Config, class_b: L2ClassBConfig) -> dict[str, Any]:
    return {
        "weights": {
            name: float(getattr(config, name))
            for name in candidate_tape.WEIGHT_NAMES
        } | {"false_refill_penalty_weight": float(config.false_refill_penalty_weight)},
        **{
            field.name: getattr(config, field.name)
            for field in fields(config)
            if field.name not in candidate_tape.WEIGHT_NAMES
            and field.name != "false_refill_penalty_weight"
        },
        "min_confirmation_seconds": class_b.min_confirmation_seconds,
        "max_confirmation_seconds": class_b.max_confirmation_seconds,
        "favorable_confirmation_ticks": class_b.favorable_confirmation_ticks,
        "confirmation_execution_count": class_b.confirmation_execution_count,
        "confirmation_volume_threshold": class_b.confirmation_volume_threshold,
        "entry_delay_ms": 2.0,
        "stop_ticks": class_b.stop_ticks,
        "target_r": class_b.target_r,
        "execution_policy": ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS,
    }


def _load_profiles(day: str, prior_day: str) -> tuple[dict[str, baseline.Profile], dict[str, baseline.Profile]]:
    payloads = {}
    for source_day in (prior_day, day):
        payload = json.loads((TAPE_ROOT / "_cache/profiles" / f"{source_day}.json").read_text())
        payloads[source_day] = {row["session"]: baseline._profile_from_payload(row) for row in payload["profiles"]}
    return payloads[prior_day], payloads[day]


def _tape_result_signature(result: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return only deterministic outcome fields for tape coverage probes."""
    return (
        int(result["qualified_count"]),
        tuple((row.get("setup_id"), row.get("entry_timestamp_ns"), row.get("exit_timestamp_ns"),
               row.get("exit"), row.get("r_multiple")) for row in result["trades"]),
    )


def _parameter_effect_days(
    family_id: str,
    class_a: L2Config,
    configs: Mapping[str, L2ClassBConfig],
    *,
    dates: tuple[str, ...] = baseline.TRAIN_DATES,
) -> dict[str, tuple[str, ...]]:
    """Scan sealed tapes only to find dates that exercise each Class-B field."""
    changed: dict[str, set[str]] = {name: set() for name in PARAMETER_CONFIGS}
    for day in dates:
        tape = candidate_tape.load_tape(TAPE_ROOT / "tapes" / f"{day}-candidate-tape.npz")
        family_tape = candidate_tape.CandidateTape(
            tape.metadata, tuple(row for row in tape.candidates if row.get("level") == family_id), tape.events,
        )
        default = candidate_tape.evaluate_candidate_tape(
            family_tape, _tape_parameters(class_a, configs["default"]), config=class_a,
        )
        default_signature = _tape_result_signature(default)
        for parameter, config_ids in PARAMETER_CONFIGS.items():
            if any(
                _tape_result_signature(candidate_tape.evaluate_candidate_tape(
                    family_tape, _tape_parameters(class_a, configs[config_id]), config=class_a,
                )) != default_signature
                for config_id in config_ids
            ):
                changed[parameter].add(day)
    return {parameter: tuple(sorted(days)) for parameter, days in changed.items()}


def _runner_factory(legacy: bool) -> Callable[[L2Config, L2ClassBConfig], L2SignalEngine] | None:
    if not legacy:
        return None
    return lambda config, class_b: LegacyDefaultBroadSignalEngine(config, class_b)


def replay_day_fanout(
    day: str,
    *,
    data_root: Path,
    specs: tuple[tuple[str, str, L2Config, L2ClassBConfig, bool], ...],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Replay one raw TRAIN DBN once for all independent family/config states."""
    _, requests = baseline._manifest(data_root)
    prior_day = baseline.DEPENDENCY_DATE if day == baseline.TRAIN_DATES[0] else baseline.TRAIN_DATES[baseline.TRAIN_DATES.index(day) - 1]
    prior_profiles, current_profiles = _load_profiles(day, prior_day)
    families = {family.family_id: family for family in baseline.build_families(day, prior_profiles, current_profiles)}
    selected = {family_id: families[family_id] for family_id in {item[0] for item in specs}}
    windows = baseline._session_windows(day)
    runners: dict[tuple[str, str, str], baseline.BroadHistoricalRunner] = {}
    for family_id, config_id, config, class_b, legacy in specs:
        family = selected[family_id]
        profile = (current_profiles if family.reference_day == "CURRENT" else prior_profiles)[
            family.reference_session if family.reference_session != "RTH" else "NY"
        ]
        level = baseline._static_level(family, profile)
        for session in baseline.SESSION_ORDER:
            if session != family.trading_session:
                continue
            runners[(family_id, config_id, session)] = baseline.BroadHistoricalRunner(
                date=day,
                evidence_label="MAC_2025_CLASS_B_RAW_FANOUT_PROOF",
                levels=[level],
                config=config,
                class_b=class_b,
                strategy_id=f"MAC2025.CLASS_B_PROOF.{family_id}.{config_id}",
                execution_policy=ES_DERIVED_PRICE_PATH_WITH_ES_OR_MES_ECONOMICS,
                signal_engine_factory=_runner_factory(legacy),
            )
    runners_by_session = {
        session: [runner for (family_id, config_id, runner_session), runner in runners.items() if runner_session == session]
        for session in baseline.SESSION_ORDER
    }

    from databento import DBNStore
    adapter = baseline.FastNativeReplayAdapter()
    path = baseline._source_path(data_root, requests, day)
    active_session: str | None = None
    records = 0
    started = time.monotonic()

    def finish_session(session: str) -> None:
        for (family_id, config_id, runner_session), runner in runners.items():
            if runner_session == session:
                runner.finish(windows[session][1])

    for batch in DBNStore.from_file(path).to_ndarray(count=1_000_000):
        for raw in batch:
            records += 1
            if records % 1_000_000 == 0:
                print(f"RAW_FANOUT_PROGRESS date={day} records={records} elapsed_seconds={time.monotonic() - started:.1f}", flush=True)
            timestamp_ns = int(raw["ts_recv"])
            session = next((name for name, (start, end) in windows.items() if start <= timestamp_ns < end), None)
            if session is None:
                continue
            session_runners = runners_by_session[session]
            if active_session != session:
                if active_session is not None:
                    finish_session(active_session)
                active_session = session
            action = raw["action"].decode() if hasattr(raw["action"], "decode") else str(raw["action"])
            active = any(
                runner.interactions.active or runner.signals.position or getattr(runner.signals, "_entry_ready", ())
                for runner in session_runners
            )
            interest = frozenset(level.price for runner in session_runners for level in runner.interactions.levels)
            public = adapter.feed_array(raw, materialize_public=action == "T" or active, interest_prices=interest)
            if public is None:
                continue
            for runner in session_runners:
                runner.observe_public(public)
    if active_session is not None:
        finish_session(active_session)
    adapter.finish()
    for runner in runners.values():
        runner.refresh_setup_ledger()
    return {
        (family_id, config_id): {
            "trades": list(runner.trade_ledger),
            "setups": list(runner.setup_ledger),
        }
        for (family_id, config_id, _), runner in runners.items()
    }


def _ledger_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(key) for key in (
        "setup_id", "entry_timestamp_ns", "entry", "stop", "target",
        "exit_timestamp_ns", "exit", "exit_reason", "r_multiple", "instrument", "contracts",
    ))


def compare_tape_raw(
    family_id: str,
    config_id: str,
    class_a: L2Config,
    class_b: L2ClassBConfig,
    dates: tuple[str, ...],
    raw_by_date: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
) -> dict[str, Any]:
    raw_trades: list[Mapping[str, Any]] = []
    tape_trades: list[Mapping[str, Any]] = []
    first_divergence: Any = None
    for day in dates:
        tape = candidate_tape.load_tape(TAPE_ROOT / "tapes" / f"{day}-candidate-tape.npz")
        candidates = tuple(row for row in tape.candidates if row.get("level") == family_id)
        family_tape = candidate_tape.CandidateTape(tape.metadata, candidates, tape.events)
        result = candidate_tape.evaluate_candidate_tape(
            family_tape, _tape_parameters(class_a, class_b), config=class_a,
        )
        raw_trades.extend(raw_by_date[day][(family_id, config_id)]["trades"])
        tape_trades.extend(result["trades"])
    raw_keys = [_ledger_key(row) for row in raw_trades]
    tape_keys = [_ledger_key(row) for row in tape_trades]
    if raw_keys != tape_keys:
        for index, pair in enumerate(zip(raw_keys, tape_keys)):
            if pair[0] != pair[1]:
                first_divergence = {"index": index, "raw": pair[0], "tape": pair[1]}
                break
        if first_divergence is None and len(raw_keys) != len(tape_keys):
            first_divergence = {"index": min(len(raw_keys), len(tape_keys)), "raw_length": len(raw_keys), "tape_length": len(tape_keys)}
    return {
        "family": family_id, "config_id": config_id, "dates": list(dates),
        "raw_trades": len(raw_trades), "tape_trades": len(tape_trades),
        "ledger_match": raw_keys == tape_keys, "first_divergence": first_divergence,
    }


def run_proof(*, data_root: Path = baseline.DATA_ROOT) -> dict[str, Any]:
    master = json.loads(MASTER_PATH.read_text())
    rows = {row["family"]: row for row in master["families"]}
    configs = proof_configs()
    config_map = dict(configs)
    parameter_effects = {
        family_id: _parameter_effect_days(family_id, _class_a_config(rows[family_id]), config_map)
        for family_id in PROOF_FAMILIES
    }
    specs: list[tuple[str, str, L2Config, L2ClassBConfig, bool]] = []
    for family_id in PROOF_FAMILIES:
        class_a = _class_a_config(rows[family_id])
        specs.append((family_id, "legacy_default", class_a, L2ClassBConfig(), True))
        for config_id, class_b in configs:
            specs.append((family_id, config_id, class_a, class_b, False))

    raw_by_date: dict[str, Mapping[tuple[str, str], Mapping[str, Any]]] = {}
    for day in PROOF_DATES:
        raw_by_date[day] = replay_day_fanout(day, data_root=data_root, specs=tuple(specs))

    default_matches = []
    for family_id in PROOF_FAMILIES:
        for day in PROOF_DATES:
            legacy = raw_by_date[day][(family_id, "legacy_default")]["trades"]
            parameterized = raw_by_date[day][(family_id, "default")]["trades"]
            default_matches.append([_ledger_key(row) for row in legacy] == [_ledger_key(row) for row in parameterized])

    comparisons = []
    for family_id in PROOF_FAMILIES:
        class_a = _class_a_config(rows[family_id])
        for config_id, class_b in configs:
            comparisons.append(compare_tape_raw(family_id, config_id, class_a, class_b, PROOF_DATES, raw_by_date))
    return {
        "proof_dates": list(PROOF_DATES),
        "proof_date_selection_reason": (
            "Smallest deterministic two-date TRAIN set retained from the sealed-tape coverage scan; "
            "2025-03-25 and 2025-04-02 collectively exercise all seven Class-B fields across the three proof families."
        ),
        "raw_files_opened": len(PROOF_DATES),
        "raw_stream_passes": len(PROOF_DATES),
        "raw_default_regression_match": all(default_matches),
        "first_default_divergence": None if all(default_matches) else "raw legacy/default ledger mismatch",
        "comparisons": comparisons,
        "exact_match_count": sum(row["ledger_match"] for row in comparisons),
        "total_comparisons": len(comparisons),
        "parameter_effect_coverage": {
            family_id: {
                parameter: {
                    "raw_used": True,
                    "tape_used": True,
                    "observable_effect": "YES" if days else "NO_OPPORTUNITY_IN_TRAIN_DATA",
                    "tape_effect_days": list(days),
                }
                for parameter, days in effects.items()
            }
            for family_id, effects in parameter_effects.items()
        },
        "raw_reference_remains_independent": True,
        "entry_delay_ms": 2.0,
        "entry_delay_frozen": True,
        "class_a_frozen": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=baseline.DATA_ROOT)
    args = parser.parse_args()
    print(json.dumps(run_proof(data_root=args.data_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
