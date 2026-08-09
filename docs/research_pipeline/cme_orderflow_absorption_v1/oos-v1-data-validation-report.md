# OOS ESU6 MBO data validation report

## Status: OOS_DATA_ACQUIRED_AND_SEALED

Target: `data/cme_orderflow_absorption_v1/oos_v1/ESU6/mbo/ESU6_2026-08-03_2026-08-08_mbo.dbn`

This is a local, read-only integrity validation. No network operation, Databento API call, data modification, book reconstruction for strategy use, interaction construction, scoring, signal/trade logic, strategy execution, or PnL calculation was performed.

## Identity and embedded metadata

| Check | Observed | Required | Result |
|---|---|---|---|
| Bytes | 1,068,425,668 | 1,068,425,668 | PASS |
| SHA-256 | `BE4B56639E56DF9AACE81621E4E276463EA8AF889104F35F1744400310D53AA3` | `BE4B56639E56DF9AACE81621E4E276463EA8AF889104F35F1744400310D53AA3` | PASS |
| DBN version | 3 | DBN parser-readable | PASS |
| Dataset / schema | `GLBX.MDP3` / `mbo` | `GLBX.MDP3` / `mbo` | PASS |
| Symbology | `raw_symbol` `ESU6` -> `42140870` | `ESU6` -> `42140870` | PASS |
| Embedded receive interval | `[2026-08-03T00:00:00Z, 2026-08-08T00:00:00Z)` | sealed DATA_REQUEST_INTERVAL | PASS |
| Observed record instrument IDs | `42140870` only | `42140870` only | PASS |

The local Databento `DBNStore` parser streamed all 61,106,259 records without materializing the file or silently dropping records. Malformed parser records: 0. Unknown side codes: 0. The 3,140 `N` actions are provider no-op/heartbeat-style MBO records (`side=N`, zero size/order ID, NaN price), not malformed records.

## Databento historical start snapshots

`DATA_REQUEST_INTERVAL` is governed by `ts_recv`, not by a universal `ts_event >= start` rule. The 9,165 records with `ts_event` before `2026-08-03T00:00:00Z` are classified as `EXPECTED_DATABENTO_MBO_START_SNAPSHOT_HISTORY`, never as coverage gaps. They are five valid F_SNAPSHOT initializations: 5 `R` resets and 9,160 `A` adds. Every one has instrument ID `42140870`, action `R` or `A`, F_SNAPSHOT set, and a deterministic snapshot-emission `ts_recv` of 00:00:00 UTC on one of the five OOS dates.

| Snapshot `ts_recv` | Pre-start `ts_event` records | Reset / adds | F_LAST completion | Result |
|---|---:|---:|---|---|
| 2026-08-03T00:00:00.000000000Z | 7,330 | 1 / 7,329 | observed | PASS |
| 2026-08-04T00:00:00.000000000Z | 844 | 1 / 843 | observed | PASS |
| 2026-08-05T00:00:00.000000000Z | 378 | 1 / 377 | observed | PASS |
| 2026-08-06T00:00:00.000000000Z | 312 | 1 / 311 | observed | PASS |
| 2026-08-07T00:00:00.000000000Z | 301 | 1 / 300 | observed | PASS |

The five snapshot groups each begin with `R`, contain only valid initialization actions, and complete with F_LAST. The first ordinary, non-snapshot incremental record follows the completed initial snapshot at `2026-08-03T00:00:00.001557027Z` (`ts_event`; its `ts_recv` is `2026-08-03T00:00:00.002583459Z`). A book is eligible only after its complete valid R/A/F_LAST snapshot; snapshots are permanently excluded from OOS interactions, scoring, signal/entry/trade eligibility, and performance eligibility. Ordinary incremental records are then causally gated by provider iterator order and `ts_recv` in the OOS interval, with `ts_event` in the OOS interval.

Any pre-start `ts_event` record without F_SNAPSHOT, a non-`R`/`A` initialization action, a wrong instrument, a non-deterministic initial snapshot `ts_recv`, an ordinary record before F_LAST, or an incomplete snapshot is repair-required. None was observed.

## Causal chronology diagnostics

`ts_recv` regressions: 0. `ts_event` regressions: 4, corresponding to the historical snapshot resets and not to ordinary incremental causality. The channel is `0` only. Sequence zero / equal / decreases / nonconsecutive increases are `5 / 10,762,485 / 1,296 / 4,030,504`; Databento/CME sequence is channel-scoped diagnostic metadata, not a global strict total-order requirement. No observed sequence or channel condition broke provider-order causal reconstruction, so none is treated as an integrity failure.

## UTC-date coverage and RTH boundary check

The five requested receive-time dates are present. `ts_event` history dated 2026-08-02 is valid snapshot initialization history, not a sixth OOS date and not a coverage gap. For each expected date, ordinary records occur immediately after the UTC-day start and immediately before the UTC-day end (except the scheduled Friday early close at 21:00 UTC). A first/last RTH boundary check for `[13:30, 20:00)` UTC finds records immediately after open and immediately before close on all five dates.

| OOS UTC date | Records with `ts_event` on date | First source event | Last source event | Executions | RTH records / first / last |
|---|---:|---|---|---:|---|
| 2026-08-03 | 10,790,417 | 00:00:00.001557027 | 23:59:59.988529853 | 1,368,401 | 7,611,026 / 13:30:00.000081813 / 19:59:59.999984489 |
| 2026-08-04 | 11,787,580 | 00:00:00.000048929 | 23:59:59.981805841 | 1,609,545 | 8,608,805 / 13:30:00.000185129 / 19:59:59.999949547 |
| 2026-08-05 | 13,387,729 | 00:00:00.000035945 | 23:59:59.950266223 | 1,538,524 | 10,658,763 / 13:30:00.000033783 / 19:59:59.999811267 |
| 2026-08-06 | 13,504,456 | 00:00:00.000041511 | 23:59:59.863162847 | 1,363,227 | 10,423,214 / 13:30:00.000022131 / 19:59:59.999907221 |
| 2026-08-07 | 11,626,912 | 00:00:00.001145633 | 21:00:00.065056825 | 1,302,967 | 9,005,071 / 13:30:00.000114541 / 19:59:59.999929317 |

No frozen strategy parameters were changed. No interactions, scores, signals, entries, trades, performance calculations, strategy execution, or PnL calculation was performed.
