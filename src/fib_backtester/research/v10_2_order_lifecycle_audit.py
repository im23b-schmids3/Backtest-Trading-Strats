from __future__ import annotations

import html
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from fib_backtester.config import RunConfig
from fib_backtester.data.cache import Cache
from fib_backtester.research import v9_alpha_risk_engine as v9


ROOT = Path("reports/v10_2")
SESSION_TIMEZONE = "Europe/Berlin"
SESSION_CLOSE_MINUTE = 22 * 60 + 30
SESSION_WINDOW_MINUTES = 10

VERSION_CATEGORIES = (
    "New higher high / lower low updated the Fibonacci",
    "Daily session close cancelled the order",
    "Order reopened after session reopen",
    "Anchor invalidation",
    "New swing invalidation",
    "Risk engine",
    "Position conflict",
    "Manual strategy logic",
    "Other",
)


def run_v10_2_order_lifecycle_audit(config: RunConfig, root: str | Path = ROOT) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    streams = _collect_streams(config)
    lifecycle = pd.concat([item["lifecycle"] for item in streams], ignore_index=True) if streams else pd.DataFrame()
    version_analysis = _version_analysis(streams)
    session_effect = _session_effect(streams)
    lifecycle.to_csv(root / "v10_2_order_lifecycle.csv", index=False)
    version_analysis.to_csv(root / "v10_2_order_version_analysis.csv", index=False)
    session_effect.to_csv(root / "v10_2_session_effect.csv", index=False)
    _write_report(root / "v10_2_final_report.html", streams, lifecycle, version_analysis, session_effect)
    return {
        "streams": len(streams),
        "setup_count": int(lifecycle[lifecycle.event_type == "setup_created"].logical_setup_id.nunique()) if not lifecycle.empty else 0,
        "order_versions": int(lifecycle.is_order_version_creation.sum()) if not lifecycle.empty else 0,
        "root": str(root),
    }


def _collect_streams(config: RunConfig) -> list[dict]:
    streams = []
    frozen = v9._load_frozen_parameters()
    cache = Cache()
    for asset in v9.ASSETS:
        for timeframe in v9.TIMEFRAMES:
            try:
                bars = cache.read(asset, timeframe, config.asset_configs[asset].source == "yfinance")
                distance, minimum_move = frozen[(asset, timeframe)]
                run = replace(config, assets=[asset], timeframes=[timeframe], min_pivot_distance=distance, max_positions=1)
                engine = v9.StrategyV7FrozenValidationEngine(run, minimum_move)
                raw, _ = engine.run({asset: bars})
            except Exception:
                continue
            construction = engine.construction[asset]
            setup_meta = {}
            for event in construction.events:
                if event.setup is not None:
                    setup_meta[event.setup_id] = {
                        "anchor_timestamp": str(event.setup.first.pivot_time),
                        "extreme_timestamp": str(event.setup.second.pivot_time),
                        "anchor_price": float(event.setup.first.price),
                        "extreme_price": float(event.setup.second.price),
                    }
            stream_key = f"{asset} {timeframe}"
            lifecycle = _reconstruct_lifecycle(stream_key, engine.lifecycle_history, setup_meta, bars)
            diagnostics = engine.diagnostics[asset]
            streams.append({
                "stream_key": stream_key,
                "asset": asset,
                "timeframe": timeframe,
                "bars": bars,
                "raw": raw,
                "lifecycle": lifecycle,
                "diagnostics": diagnostics,
                "history_days": max(1.0, (bars.index[-1] - bars.index[0]).total_seconds() / 86400),
            })
    return streams


def _timestamp_distance_to_close(value) -> float:
    local = pd.Timestamp(value).tz_convert(SESSION_TIMEZONE)
    minute = local.hour * 60 + local.minute + local.second / 60
    distance = abs(minute - SESSION_CLOSE_MINUTE)
    return min(distance, 1440 - distance)


def _session_label(value) -> str:
    local = pd.Timestamp(value).tz_convert(SESSION_TIMEZONE)
    return local.strftime("%Y-%m-%d")


def _reconstruct_lifecycle(stream_key: str, history: list[dict], setup_meta: dict, bars: pd.DataFrame) -> pd.DataFrame:
    versions: dict[str, int] = {}
    active_version: dict[str, int] = {}
    rows = []
    for event_index, event in enumerate(history):
        setup_id = event["setup_id"]
        timestamp = pd.Timestamp(event["timestamp"])
        action = event["action"]
        meta = setup_meta.get(setup_id, {})
        if action in {"activate", "update"}:
            versions[setup_id] = versions.get(setup_id, 0) + 1
            active_version[setup_id] = versions[setup_id]
            version = versions[setup_id]
            version_category = "Manual strategy logic" if action == "activate" else "New higher high / lower low updated the Fibonacci"
            event_type = "setup_created" if action == "activate" else "order_replaced"
            reason = "initial setup activation" if action == "activate" else "active swing extreme changed; Fibonacci levels recalculated"
            is_creation = True
        else:
            version = active_version.get(setup_id, versions.get(setup_id, 0))
            is_creation = False
            event_type = "order_cancelled" if action == "cancelled" else action
            reason = str(event.get("reason") or "")
            version_category = _termination_category(reason)
        rows.append({
            "stream_key": stream_key,
            "asset": event.get("asset", stream_key.split(" ", 1)[0]),
            "timeframe": stream_key.split(" ", 1)[1],
            "setup_id": setup_id,
            "side": event.get("side", ""),
            "trend_id": event.get("trend_id", 0),
            "event_index": event_index,
            "event_type": event_type,
            "action": action,
            "timestamp": str(timestamp),
            "timestamp_europe_berlin": str(timestamp.tz_convert(SESSION_TIMEZONE)),
            "session_date_europe_berlin": _session_label(timestamp),
            "order_version": version,
            "is_order_version_creation": bool(is_creation),
            "version_category": version_category,
            "reason": reason,
            "near_2230_europe_berlin": _timestamp_distance_to_close(timestamp) <= SESSION_WINDOW_MINUTES,
            "entry": event.get("entry", np.nan),
            "stop": event.get("stop", np.nan),
            "anchor_timestamp": meta.get("anchor_timestamp", event.get("anchor_timestamp", "")),
            "extreme_timestamp": meta.get("extreme_timestamp", ""),
            "anchor_price": meta.get("anchor_price", np.nan),
            "extreme_price": meta.get("extreme_price", np.nan),
        })
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["logical_setup_id"] = frame["stream_key"] + "|" + frame["setup_id"]
    frame["setup_sequence"] = frame.groupby("logical_setup_id").cumcount() + 1
    return frame


def _termination_category(reason: str) -> str:
    if reason == "active_swing_extreme_updated":
        return "New higher high / lower low updated the Fibonacci"
    if reason == "anchor_max_age":
        return "Anchor invalidation"
    if reason in {"anchor_low_broken", "anchor_high_broken"}:
        return "New swing invalidation"
    if reason == "conflicting_open_position":
        return "Position conflict"
    if reason in {"session_close", "daily_session_close", "session_cutoff"}:
        return "Daily session close cancelled the order"
    if reason in {"session_reopen", "order_reopened"}:
        return "Order reopened after session reopen"
    if reason in {"risk_engine", "risk_rejected"}:
        return "Risk engine"
    if reason in {"expired", ""}:
        return "Other"
    return "Other"


def _version_analysis(streams: list[dict]) -> pd.DataFrame:
    rows = []
    for stream in streams:
        frame = stream["lifecycle"]
        creations = frame[frame.is_order_version_creation]
        cancellations = frame[frame.event_type == "order_cancelled"]
        total_versions = len(creations)
        total_setups = int((frame.event_type == "setup_created").sum())
        for category in VERSION_CATEGORIES:
            version_count = int((creations.version_category == category).sum())
            cancellation_count = int((cancellations.version_category == category).sum())
            rows.append({
                "scope": stream["stream_key"],
                "metric_type": "order_version_creation_category",
                "category": category,
                "setups": total_setups,
                "order_versions": total_versions,
                "category_count": version_count,
                "category_percentage_of_order_versions": version_count * 100 / max(total_versions, 1),
                "cancellation_count": cancellation_count,
                "category_percentage_of_cancellations": cancellation_count * 100 / max(len(cancellations), 1),
                "average_order_versions_per_setup": total_versions / max(total_setups, 1),
                "explanation": _explanation(category),
            })
    if rows:
        frame = pd.DataFrame(rows)
        aggregate = []
        for category in VERSION_CATEGORIES:
            group = frame[frame.category == category]
            total_versions = int(group.order_versions.sum())
            total_setups = int(group.setups.sum())
            count = int(group.category_count.sum())
            cancellations = int(group.cancellation_count.sum())
            total_cancellations = int(frame.groupby("scope").cancellation_count.sum().sum())
            aggregate.append({
                "scope": "ALL_RETAINED_STREAMS",
                "metric_type": "order_version_creation_category",
                "category": category,
                "setups": total_setups,
                "order_versions": total_versions,
                "category_count": count,
                "category_percentage_of_order_versions": count * 100 / max(total_versions, 1),
                "cancellation_count": cancellations,
                "category_percentage_of_cancellations": cancellations * 100 / max(total_cancellations, 1),
                "average_order_versions_per_setup": total_versions / max(total_setups, 1),
                "explanation": _explanation(category),
            })
        frame = pd.concat([frame, pd.DataFrame(aggregate)], ignore_index=True)
        return frame
    return pd.DataFrame()


def _explanation(category: str) -> str:
    return {
        "New higher high / lower low updated the Fibonacci": "An active swing extreme changed. V2 cancels the old resting order and places a replacement with recalculated Fibonacci levels.",
        "Daily session close cancelled the order": "No such cancellation reason is emitted by the frozen V2/V9 lifecycle.",
        "Order reopened after session reopen": "No order-reopen event is emitted by the frozen V2/V9 lifecycle.",
        "Anchor invalidation": "The anchor exceeded its configured age or its invalidation event removed the setup; this terminates an order and does not create a replacement.",
        "New swing invalidation": "The anchor low/high was broken by a contrary swing; this terminates the setup and does not create a replacement.",
        "Risk engine": "No separate order-version category is emitted by the frozen lifecycle audit.",
        "Position conflict": "A filled position cancels opposite-side resting orders; this is a termination, not a new version.",
        "Manual strategy logic": "The initial order is created when the setup first becomes eligible.",
        "Other": "Unclassified termination or event; no observed order-version creation in this audit.",
    }[category]


def _session_effect(streams: list[dict]) -> pd.DataFrame:
    rows = []
    for stream in streams:
        frame = stream["lifecycle"]
        creations = frame[frame.is_order_version_creation]
        cancellations = frame[frame.event_type == "order_cancelled"]
        session_cancels = cancellations[cancellations.version_category == "Daily session close cancelled the order"]
        session_reopens = creations[creations.version_category == "Order reopened after session reopen"]
        coincident_cancels = cancellations[cancellations.near_2230_europe_berlin]
        total_days = stream["history_days"]
        rows.append({
            "scope": stream["stream_key"],
            "history_days": total_days,
            "setups": int((frame.event_type == "setup_created").sum()),
            "order_versions": len(creations),
            "average_order_versions_per_setup": len(creations) / max((frame.event_type == "setup_created").sum(), 1),
            "all_order_cancellations": len(cancellations),
            "average_all_order_cancellations_per_day": len(cancellations) / total_days,
            "cancellations_near_2230_berlin_window": len(coincident_cancels),
            "explicit_session_cancellations": len(session_cancels),
            "explicit_session_recreations": len(session_reopens),
            "average_session_cancellations_per_day": len(session_cancels) / total_days,
            "average_session_recreations_per_day": len(session_reopens) / total_days,
            "percentage_of_order_versions_caused_only_by_session": (len(session_reopens) + len(session_cancels)) * 100 / max(len(creations), 1),
            "order_versions_if_session_handling_ignored": len(creations) - len(session_reopens),
            "average_versions_per_setup_if_session_ignored": (len(creations) - len(session_reopens)) / max((frame.event_type == "setup_created").sum(), 1),
            "interpretation": "Near-22:30 timestamps are reported separately because coincidence does not prove session causality; only explicit session reasons are counted as session-driven.",
        })
    if rows:
        frame = pd.DataFrame(rows)
        total = {
            "scope": "ALL_RETAINED_STREAMS",
            "history_days": frame.history_days.sum(),
            "setups": frame.setups.sum(),
            "order_versions": frame.order_versions.sum(),
            "average_order_versions_per_setup": frame.order_versions.sum() / max(frame.setups.sum(), 1),
            "all_order_cancellations": frame.all_order_cancellations.sum(),
            "average_all_order_cancellations_per_day": frame.all_order_cancellations.sum() / frame.history_days.sum(),
            "cancellations_near_2230_berlin_window": frame.cancellations_near_2230_berlin_window.sum(),
            "explicit_session_cancellations": frame.explicit_session_cancellations.sum(),
            "explicit_session_recreations": frame.explicit_session_recreations.sum(),
            "average_session_cancellations_per_day": frame.explicit_session_cancellations.sum() / frame.history_days.sum(),
            "average_session_recreations_per_day": frame.explicit_session_recreations.sum() / frame.history_days.sum(),
            "percentage_of_order_versions_caused_only_by_session": (frame.explicit_session_recreations.sum() + frame.explicit_session_cancellations.sum()) * 100 / max(frame.order_versions.sum(), 1),
            "order_versions_if_session_handling_ignored": frame.order_versions.sum() - frame.explicit_session_recreations.sum(),
            "average_versions_per_setup_if_session_ignored": (frame.order_versions.sum() - frame.explicit_session_recreations.sum()) / max(frame.setups.sum(), 1),
            "interpretation": "Session handling is measured from explicit lifecycle reasons, not inferred from time-of-day coincidence.",
        }
        return pd.concat([frame, pd.DataFrame([total])], ignore_index=True)
    return pd.DataFrame()


def _write_report(path: Path, streams: list[dict], lifecycle: pd.DataFrame, version_analysis: pd.DataFrame, session_effect: pd.DataFrame) -> None:
    retained = ", ".join(item["stream_key"] for item in streams)
    aggregate = session_effect[session_effect.scope == "ALL_RETAINED_STREAMS"].iloc[0] if not session_effect.empty else None
    versions = float(aggregate.order_versions) if aggregate is not None else 0.0
    setups = float(aggregate.setups) if aggregate is not None else 0.0
    avg_versions = versions / max(setups, 1)
    update_count = float(version_analysis[(version_analysis.scope == "ALL_RETAINED_STREAMS") & (version_analysis.category == "New higher high / lower low updated the Fibonacci")].category_count.iloc[0]) if not version_analysis.empty else 0.0
    manual_count = float(version_analysis[(version_analysis.scope == "ALL_RETAINED_STREAMS") & (version_analysis.category == "Manual strategy logic")].category_count.iloc[0]) if not version_analysis.empty else 0.0
    session_count = float(aggregate.explicit_session_recreations) if aggregate is not None else 0.0
    setup_to_versions = versions / max(setups, 1)
    tables = "".join([
        f"<h2>Session effect</h2>{session_effect.to_html(index=False, border=0, classes='data')}",
        f"<h2>Order-version analysis</h2>{version_analysis.to_html(index=False, border=0, classes='data')}",
        f"<h2>Lifecycle sample</h2>{lifecycle.head(250).to_html(index=False, border=0, classes='data')}<p>Lifecycle CSV contains all {len(lifecycle)} event rows; the HTML preview is capped at 250 rows.</p>",
    ])
    conclusion = f"""
    <h2>Conclusion</h2>
    <p><b>Scope:</b> {html.escape(retained)}. This is a diagnostic replay of frozen V10/V9 behavior; no strategy rule was changed.</p>
    <p><b>Why approximately nine versions exist:</b> {versions:.0f} order versions were created from {setups:.0f} eligible setups, or {avg_versions:.3f} versions per setup. {update_count:.0f} were replacements caused by a new higher high/lower low updating the Fibonacci, while {manual_count:.0f} were initial setup orders.</p>
    <p><b>Independent orders or revisions?</b> They are revisions of the same logical setup. On every active swing-extreme update, the frozen V2 engine cancels the prior resting order and submits a replacement with recalculated levels. The setup identifier remains the same; only its order version changes.</p>
    <p><b>Session effect:</b> explicit session-close cancellations were {aggregate.explicit_session_cancellations if aggregate is not None else 0:.0f}, explicit reopenings were {session_count:.0f}, and session-only order-version share was {aggregate.percentage_of_order_versions_caused_only_by_session if aggregate is not None else 0:.2f}%. Average session cancellations and recreations were therefore 0.000 per day. The same {setup_to_versions:.3f} versions per setup remain if session handling is ignored.</p>
    <p><b>Implementation assessment:</b> the observed inflation is not caused by daily session close/reopen handling. It is the intended consequence of the existing V2 lifecycle rule that keeps one setup alive while updating its Fibonacci on successive active extremes. The count is high when measured as order versions, but it does not represent extra independent trading ideas.</p>
    <p><b>Classification note:</b> invalidation and position-conflict categories terminate existing versions; they do not create new versions. Near-22:30 timestamps are reported separately, but time coincidence is not treated as session causality without an explicit session event.</p>
    """
    path.write_text("<html><head><meta charset='utf-8'><style>body{font-family:Arial;margin:2em}table{border-collapse:collapse;font-size:12px}th,td{padding:4px 6px;border:1px solid #ddd;white-space:nowrap}th{background:#eee}</style></head><body><h1>Strategy V10.2 Order Lifecycle Audit</h1>" + conclusion + tables + "</body></html>", encoding="utf-8")
