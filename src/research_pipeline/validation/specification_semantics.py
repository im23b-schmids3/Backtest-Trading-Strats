from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field

from ..schemas.strategy_spec import StrictModel


class SpecificationValidationIssue(StrictModel):
    error_code: str
    field_path: str
    received_value: Any = None
    expected_constraint: str
    explanation: str
    repair_hint: str


class SpecificationValidationReport(StrictModel):
    valid: bool
    pydantic_valid: bool
    semantic_valid: bool
    errors: list[SpecificationValidationIssue] = Field(default_factory=list)
    blocking_ambiguities: list[str] = Field(default_factory=list)
    normalized_payload_hash: str | None = None


class SpecificationProvenance(StrictModel):
    confirmed: list[str] = Field(default_factory=list)
    technical_translations: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    blocking_ambiguities: list[str] = Field(default_factory=list)
    source_intake_hash: str | None = None


class CanonicalSpecification(StrictModel):
    payload: dict[str, Any]
    provenance: SpecificationProvenance
    normalized_hash: str


_TIMEFRAME_RE = re.compile(r"^(?:\d+)(?:m|h|d|w)$", re.IGNORECASE)
_MARKET_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,31}$")
_PERCENT_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*%\s*$")
_CAPITAL_RE = re.compile(r"^\s*[$€£]?\s*([\d,]+(?:\.\d+)?)\s*$")


def _text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value or "")


def _normalize_scalar(value: Any, *, key: str) -> Any:
    if isinstance(value, str):
        percent = _PERCENT_RE.fullmatch(value)
        if percent and any(token in key.lower() for token in ("fraction", "percent", "percentage", "rate")):
            return float(Decimal(percent.group(1)) / Decimal("100"))
        if key.lower() in {"initial_cash", "capital", "initial_capital", "account_size"}:
            capital = _CAPITAL_RE.fullmatch(value)
            if capital:
                parsed = Decimal(capital.group(1).replace(",", ""))
                return int(parsed) if parsed == parsed.to_integral() else float(parsed)
        return " ".join(value.strip().split())
    return value


def _normalize_date(value: Any, field_path: str, issues: list[SpecificationValidationIssue]) -> Any:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            try:
                return date.fromisoformat(value).isoformat()
            except ValueError:
                issues.append(SpecificationValidationIssue(error_code="INVALID_DATE", field_path=field_path,
                    received_value=value, expected_constraint="ISO-8601 date or timestamp", explanation="Date boundaries must be valid ISO dates.",
                    repair_hint="Use YYYY-MM-DD and make the end boundary exclusive."))
                return value
    issues.append(SpecificationValidationIssue(error_code="INVALID_DATE_TYPE", field_path=field_path,
        received_value=value, expected_constraint="ISO-8601 date or timestamp", explanation="Date boundaries must be strings or date-like values.",
        repair_hint="Replace the value with an ISO-8601 date such as 2025-01-01."))
    return value


def normalize_specification_payload(raw: dict[str, Any], provenance: SpecificationProvenance | None = None) -> CanonicalSpecification:
    """Normalize only representation-level differences before strict validation."""
    issues: list[SpecificationValidationIssue] = []
    data = deepcopy(raw)
    for key in ("strategy_id", "version", "name", "description", "hypothesis", "strategy_family", "entry_logic", "initial_stop_logic", "exit_logic"):
        if isinstance(data.get(key), str):
            data[key] = " ".join(data[key].strip().split())
    for key in ("markets", "timeframes"):
        values = data.get(key)
        if isinstance(values, list):
            normalized = [str(item).strip().upper() if key == "markets" else str(item).strip().lower() for item in values]
            data[key] = sorted(dict.fromkeys(normalized))
    for key in ("long_rules", "short_rules", "session_assumptions", "invariants", "required_data", "known_limitations"):
        if isinstance(data.get(key), list):
            data[key] = [" ".join(str(item).strip().split()) for item in data[key]]
    params = data.get("baseline_parameters")
    if isinstance(params, dict):
        normalized_params: dict[str, Any] = {}
        for key, value in params.items():
            normalized_key = str(key).strip()
            normalized_value = _normalize_scalar(value, key=normalized_key)
            if normalized_key.lower() in {"test_start_date", "test_end_date", "start_date", "end_date"}:
                normalized_value = _normalize_date(normalized_value, f"baseline_parameters.{normalized_key}", issues)
            if normalized_key.lower() in {"session_timezone", "timezone", "iana_timezone"} and isinstance(normalized_value, str):
                normalized_value = normalized_value.strip()
            normalized_params[normalized_key] = normalized_value
        data["baseline_parameters"] = dict(sorted(normalized_params.items()))
    families = data.get("parameter_families")
    if isinstance(families, list):
        normalized_families = []
        for family in families:
            item = deepcopy(family)
            if isinstance(item, dict):
                for key in ("name", "description", "value_type", "hypothesis_relevance"):
                    if isinstance(item.get(key), str):
                        item[key] = " ".join(item[key].strip().split())
            normalized_families.append(item)
        data["parameter_families"] = sorted(normalized_families, key=lambda item: (item.get("optimization_order", 0), item.get("name", "")) if isinstance(item, dict) else (0, ""))
    data["specification_hash"] = "pending"
    provenance = provenance or SpecificationProvenance()
    encoded = json.dumps({"payload": data, "provenance": provenance.model_dump(mode="json")}, sort_keys=True, separators=(",", ":"), default=str).encode()
    import hashlib
    return CanonicalSpecification(payload=data, provenance=provenance, normalized_hash=hashlib.sha256(encoded).hexdigest())


def semantic_validate(raw: dict[str, Any], *, provenance: SpecificationProvenance | None = None, allowed_timeframes: set[str] | None = None) -> tuple[dict[str, Any], SpecificationValidationReport, SpecificationProvenance]:
    canonical = normalize_specification_payload(raw, provenance)
    data = canonical.payload
    errors: list[SpecificationValidationIssue] = []
    provenance = canonical.provenance
    markets = data.get("markets", [])
    timeframes = data.get("timeframes", [])
    if not isinstance(markets, list) or any(not isinstance(item, str) or not _MARKET_RE.fullmatch(item) for item in markets):
        errors.append(SpecificationValidationIssue(error_code="INVALID_MARKET_IDENTIFIER", field_path="markets", received_value=markets,
            expected_constraint="uppercase repository-safe market identifiers", explanation="Market identifiers must be explicit and filesystem-safe.", repair_hint="Use symbols such as SPY, SPX, BTCUSDT, or TEST."))
    allowed = allowed_timeframes or {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"}
    if not isinstance(timeframes, list) or any(not isinstance(item, str) or not _TIMEFRAME_RE.fullmatch(item) or item not in allowed for item in timeframes):
        errors.append(SpecificationValidationIssue(error_code="INVALID_TIMEFRAME", field_path="timeframes", received_value=timeframes,
            expected_constraint=f"one or more configured timeframes: {sorted(allowed)}", explanation="Timeframes must be configured repository identifiers.", repair_hint="Use an allowed timeframe and do not invent a dataset."))
    params = data.get("baseline_parameters") if isinstance(data.get("baseline_parameters"), dict) else {}
    start = params.get("test_start_date", params.get("start_date")); end = params.get("test_end_date", params.get("end_date"))
    if start and end and str(end) <= str(start):
        errors.append(SpecificationValidationIssue(error_code="DATE_ORDER_INVALID", field_path="baseline_parameters.test_end_date", received_value=end,
            expected_constraint="end date must be after start date; end is exclusive", explanation="The chronological test interval is empty or reversed.", repair_hint="Set the exclusive end date after the start date."))
    timezone_name = params.get("session_timezone", params.get("timezone"))
    if timezone_name:
        try:
            ZoneInfo(str(timezone_name))
        except (ZoneInfoNotFoundError, ValueError):
            errors.append(SpecificationValidationIssue(error_code="INVALID_TIMEZONE", field_path="baseline_parameters.session_timezone", received_value=timezone_name,
                expected_constraint="valid IANA timezone", explanation="Session times require an IANA timezone so daylight-saving transitions are represented.", repair_hint="Use a value such as America/New_York."))
    combined = " ".join(_text(data.get(key)) for key in ("description", "hypothesis", "long_rules", "short_rules", "entry_logic", "initial_stop_logic", "exit_logic", "session_assumptions", "invariants", "known_limitations", "required_data"))
    lower = combined.lower()
    if "spy" in {str(item).lower() for item in markets} and not ("proxy" in lower and ("mapping" in lower or "phase d" in lower)):
        errors.append(SpecificationValidationIssue(error_code="PROXY_DISCLOSURE_MISSING", field_path="known_limitations", received_value=data.get("known_limitations"),
            expected_constraint="explicit SPY proxy and unsupported Phase D mapping disclosure", explanation="SPY must remain an approval-visible repository proxy and cannot imply an invented futures mapping.", repair_hint="State the proxy substitution and that existing Phase D futures mappings do not support SPY."))
    claims_exact_subhour = ("10-minute" in lower or "10 minutes" in lower or "600 seconds" in lower) and not any(token in lower for token in ("not exact 10-minute", "not an exact 10-minute", "no exact 10-minute", "not exact 10 minutes", "no exact 10 minutes", "does not claim", "cannot establish", "not claim or approximate", "not claimed", "not approximated"))
    if claims_exact_subhour and any(str(item).lower() in {"1h", "4h", "1d", "1w"} for item in timeframes):
        errors.append(SpecificationValidationIssue(error_code="INCOMPATIBLE_HOLDING_PERIOD_DATA", field_path="exit_logic", received_value=data.get("exit_logic"),
            expected_constraint="hourly data cannot claim an exact 10-minute exit", explanation="The requested holding period is finer than the declared dataset timeframe.", repair_hint="Use compatible minute data or explicitly label a same-hour repository-compatible variant."))
    if "america/new_york" in lower and ("fixed utc" in lower or "session_open_utc" in lower or "utc time" in lower):
        errors.append(SpecificationValidationIssue(error_code="FIXED_UTC_SESSION", field_path="session_assumptions", received_value=data.get("session_assumptions"),
            expected_constraint="resolve local session time using IANA timezone", explanation="A fixed UTC conversion would fail daylight-saving transitions.", repair_hint="Declare the local session time and America/New_York; convert timestamps at runtime."))
    has_negative_seed = any(token in lower for token in ("without a seed", "without seed", "no seed", "unseeded", "non-deterministic"))
    if any(token in lower for token in ("random", "coin flip", "random direction")) and (has_negative_seed or not any(token in lower for token in ("deterministic", "seed", "seeded"))):
        errors.append(SpecificationValidationIssue(error_code="NONDETERMINISTIC_RANDOMNESS", field_path="entry_logic", received_value=data.get("entry_logic"),
            expected_constraint="random direction must declare a deterministic seed rule", explanation="Randomized direction without a seed is not reproducible.", repair_hint="Declare a deterministic seed derived from the strategy and trading date."))
    if any(token in lower for token in ("every trading day", "one trade per trading day", "exactly one trade per day")) and not any(token in lower for token in ("one trade", "exactly one")):
        errors.append(SpecificationValidationIssue(error_code="TRADE_FREQUENCY_UNSPECIFIED", field_path="invariants", received_value=data.get("invariants"),
            expected_constraint="one-trade-per-day rule must be explicit", explanation="The intake promises daily participation without an explicit one-trade rule.", repair_hint="State exactly one trade per valid trading day."))
    if "5%" in lower or "5 percent" in lower:
        if "current equity" in lower or "equity" in lower:
            fraction = params.get("equity_fraction", params.get("allocation_fraction"))
            risk_language = any(token in lower for token in ("risk per trade", "risk-per-trade"))
            explicitly_not_risk = any(token in lower for token in ("not risk per trade", "not risk-per-trade", "not a risk per trade", "not a risk-per-trade"))
            if fraction != 0.05 or (risk_language and not explicitly_not_risk):
                errors.append(SpecificationValidationIssue(error_code="EQUITY_ALLOCATION_MISREPRESENTED", field_path="baseline_parameters.equity_fraction", received_value=fraction,
                    expected_constraint="5% of current equity represented as an allocation fraction", explanation="Equity allocation must not be silently converted into risk-per-trade sizing.", repair_hint="Set equity_fraction to 0.05 and describe allocation from current equity."))
    if "no stop" in lower and any(token in lower for token in ("default stop", "stop loss", "stop_distance")):
        errors.append(SpecificationValidationIssue(error_code="INVENTED_STOP", field_path="initial_stop_logic", received_value=data.get("initial_stop_logic"),
            expected_constraint="no-stop rule must remain immutable", explanation="A stop was introduced despite the confirmed no-stop rule.", repair_hint="Use explicit no-stop language and remove stop parameters."))
    if "no target" in lower and any(token in lower for token in ("default target", "profit target", "target_distance")):
        errors.append(SpecificationValidationIssue(error_code="INVENTED_TARGET", field_path="exit_logic", received_value=data.get("exit_logic"),
            expected_constraint="no-target rule must remain immutable", explanation="A target was introduced despite the confirmed no-target rule.", repair_hint="Use explicit no-target language and remove target parameters."))
    if "no optimization" in lower or "do not optimize" in lower:
        mutable = [item.get("name") for item in data.get("parameter_families", []) if isinstance(item, dict) and item.get("mutable")]
        if mutable or "grid" in lower or "optimization range" in lower:
            errors.append(SpecificationValidationIssue(error_code="INVENTED_OPTIMIZATION", field_path="parameter_families", received_value=mutable,
                expected_constraint="no optimization means no invented mutable grid", explanation="The generated specification adds research freedom not present in the intake.", repair_hint="Remove invented parameter families or mark only explicitly requested families mutable."))
    if provenance.blocking_ambiguities or provenance.missing_information:
        errors.append(SpecificationValidationIssue(error_code="BLOCKING_AMBIGUITY", field_path="provenance.blocking_ambiguities", received_value=provenance.blocking_ambiguities or provenance.missing_information,
            expected_constraint="no unresolved material ambiguity before approval", explanation="Approval cannot proceed while material intent is unresolved.", repair_hint="Ask for clarification or preserve the ambiguity and stop before approval."))
    report = SpecificationValidationReport(valid=not errors, pydantic_valid=False, semantic_valid=not errors, errors=errors,
        blocking_ambiguities=[*provenance.blocking_ambiguities, *provenance.missing_information], normalized_payload_hash=canonical.normalized_hash)
    return data, report, provenance
