# L2 V3 native MBP-10 executable-book diagnosis

This is source/adapter evidence only. No interaction, signal, confirmation,
entry, exit, PnL, or fresh-result artifact was produced.

## First prior failure

The first record rejected by the original adapter was record `5,481,627` from
`ESU6_2026-08-10_130000_224501_mbp10.dbn.zst`.

| field | value |
|---|---|
| receive UTC | 2026-08-10T21:45:30.003482Z |
| `ts_event` | 1786398330002539377 |
| `ts_recv` | 1786398330003482495 |
| action / side | `A` / `A` |
| price / size | 7777000000000 / 1 |
| flags / sequence | 128 / 7879274 |
| top bid | 7777000000000 / 3 / 2 |
| top ask | 7777000000000 / 2 / 2 |

The record temporarily locks the displayed top book (`bid_px_00 == ask_px_00`)
during the scheduled 21:00–22:00 UTC Globex maintenance/pause interval. It is
an ordinary aggregate add, not a trade, malformed fixed-point price, or MBO
reconstruction issue. The preceding record at 21:45:30.003156Z had a normal
top book (`7777.00 / 7777.50`). The subsequent cancellation/update records
remain locked through the pause. The first subsequent uncrossed two-sided book
is record `5,481,841`, received `2026-08-10T22:00:00.003885Z`, action/side
`C`/`N`, sequence `7880074`, with `bid_px_00=7776500000000` and
`ask_px_00=7778250000000`. The original adapter incorrectly required every
MBP-10 record to be executable.

## Repair

The direct native adapter now exposes four deterministic states:

- `UNINITIALIZED`: accepts records but cannot publish strategy events.
- `EXECUTABLE`: only a valid uncrossed two-sided BBO may reach the runner.
- `NON_EXECUTABLE_EXPECTED`: scheduled maintenance/clear; all executable
  quotes are cleared and no market event is published.
- `WAITING_FOR_REOPEN_BOOK`: a control/trade record lacked a complete book;
  no stale quote is retained until a valid two-sided book returns.

Ordinary one-sided/crossed add/cancel/modify records outside an expected
non-executable state still fail closed. A native trade or reset record without
a complete book is not falsely treated as corruption, but cannot produce an
execution or confirmation until a valid native book reappears.
