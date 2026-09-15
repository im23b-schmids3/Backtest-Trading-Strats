# CME L2 structural-level candidate universe

This is a candidate inventory, not a backtest, result, or production strategy
selection. The corresponding explicit manifest is
`examples/research_pipeline/cme_l2_candidate_universe.example.yaml`; it runs
only if a user explicitly passes it to `cme-l2-research` together with a
period manifest whose causal tapes already contain the named target-session
levels.

All rows use the existing absorption interaction engine. Its direction rule is
unchanged: `BUYER_ABSORPTION` is a long reversal and `SELLER_ABSORPTION` is a
short reversal. A high/VAH/resistance or low/VAL/support reference does not
introduce a separate directional rule.

## Canonical chronology used

- Asia signals: `[00:00, 08:00)` UTC; its profile is complete only at 08:00
  UTC.
- Europe signals: `[08:00, 16:30)` Europe/London, i.e. 07:00–15:30 UTC in
  summer and 08:00–16:30 UTC in winter. The session profile completes at its
  16:30 London signal end.
- NY RTH signals: `[13:30, 20:00)` UTC. Prior-RTH profile levels are computed
  from the immediately preceding completed RTH.

The source definitions are [Asia replay](/Users/sandro/Documents/Trading-Bot-Fib/Backtest-Trading-Strats/src/research_pipeline/cme_orderflow_absorption_l2_v1/asia_w04_replay.py), [Europe replay](/Users/sandro/Documents/Trading-Bot-Fib/Backtest-Trading-Strats/src/research_pipeline/cme_orderflow_absorption_l2_v1/europe_w04_replay.py), and [canonical RTH analysis](/Users/sandro/Documents/Trading-Bot-Fib/Backtest-Trading-Strats/src/research_pipeline/cme_orderflow_absorption_v1/analysis.py).

## SAME_SESSION

| Strategy ID | Target | Source | Level | Causal validity | Existing implementation | Resolver required |
|---|---|---|---|---|---|---|
| EU_PRIOR_EU_HIGH | Europe | Europe | prior high | Valid: prior completed Europe profile | Yes | No |
| EU_PRIOR_EU_LOW | Europe | Europe | prior low | Valid: prior completed Europe profile | Yes | No |
| EU_PRIOR_EU_POC | Europe | Europe | prior POC | Valid: prior completed Europe profile | Yes | No |
| EU_PRIOR_EU_VAH | Europe | Europe | prior VAH | Valid: prior completed Europe profile | Yes | No |
| EU_PRIOR_EU_VAL | Europe | Europe | prior VAL | Valid: prior completed Europe profile | Yes | No |
| EU_CURRENT_EU_HIGH | Europe | Europe | current high sweep | Valid: event-time extremum | Yes | No |
| EU_CURRENT_EU_LOW | Europe | Europe | current low sweep | Valid: event-time extremum | Yes | No |
| NY_PRIOR_NY_HIGH | NY RTH | NY RTH | prior high | Valid: prior completed RTH profile | Yes | No |
| NY_PRIOR_NY_LOW | NY RTH | NY RTH | prior low | Valid: prior completed RTH profile | Yes | No |
| NY_PRIOR_NY_POC | NY RTH | NY RTH | prior POC | Valid: prior completed RTH profile | Yes | No |
| NY_PRIOR_NY_VAH | NY RTH | NY RTH | prior VAH | Valid: prior completed RTH profile | Yes | No |
| NY_PRIOR_NY_VAL | NY RTH | NY RTH | prior VAL | Valid: prior completed RTH profile | Yes | No |
| NY_CURRENT_NY_HIGH | NY RTH | NY RTH | current high sweep | Valid: event-time extremum | Yes | No |
| NY_CURRENT_NY_LOW | NY RTH | NY RTH | current low sweep | Valid: event-time extremum | Yes | No |
| ASIA_PRIOR_ASIA_HIGH | Asia | Asia | prior high | Valid: prior completed Asia profile | Yes | No |
| ASIA_PRIOR_ASIA_LOW | Asia | Asia | prior low | Valid: prior completed Asia profile | Yes | No |
| ASIA_PRIOR_ASIA_POC | Asia | Asia | prior POC | Valid: prior completed Asia profile | Yes | No |
| ASIA_PRIOR_ASIA_VAH | Asia | Asia | prior VAH | Valid: prior completed Asia profile | Yes | No |
| ASIA_PRIOR_ASIA_VAL | Asia | Asia | prior VAL | Valid: prior completed Asia profile | Yes | No |
| ASIA_CURRENT_ASIA_HIGH | Asia | Asia | current high sweep | Valid: event-time extremum | Yes | No |
| ASIA_CURRENT_ASIA_LOW | Asia | Asia | current low sweep | Valid: event-time extremum | Yes | No |

## CROSS_SESSION_VALID

| Strategy ID | Target | Source | Level | Causal validity | Existing implementation | Resolver required |
|---|---|---|---|---|---|---|
| EU_PRIOR_NY_HIGH | Europe | prior NY RTH | high | NY completes before the following Europe open | Shared resolver | Yes |
| EU_PRIOR_NY_LOW | Europe | prior NY RTH | low | NY completes before the following Europe open | Shared resolver | Yes |
| EU_PRIOR_NY_POC | Europe | prior NY RTH | POC | NY completes before the following Europe open | Shared resolver | Yes |
| EU_PRIOR_NY_VAH | Europe | prior NY RTH | VAH | NY completes before the following Europe open | Shared resolver | Yes |
| EU_PRIOR_NY_VAL | Europe | prior NY RTH | VAL | NY completes before the following Europe open | Shared resolver | Yes |
| NY_CURRENT_EU_HIGH | NY RTH | current Europe | high sweep | Event-time high during the NY-Europe overlap; final after Europe closes | Shared resolver | Yes |
| NY_CURRENT_EU_LOW | NY RTH | current Europe | low sweep | Event-time low during the NY-Europe overlap; final after Europe closes | Shared resolver | Yes |
| NY_COMPLETED_EU_POC | NY RTH | target-date completed Europe | POC | Only after Europe’s 16:30 London profile close | Shared resolver | Yes |
| NY_COMPLETED_EU_VAH | NY RTH | target-date completed Europe | VAH | Only after Europe’s 16:30 London profile close | Shared resolver | Yes |
| NY_COMPLETED_EU_VAL | NY RTH | target-date completed Europe | VAL | Only after Europe’s 16:30 London profile close | Shared resolver | Yes |
| NY_PRIOR_EU_HIGH | NY RTH | prior Europe | high | Prior completed Europe profile is known before NY opens | Shared resolver | Yes |
| NY_PRIOR_EU_LOW | NY RTH | prior Europe | low | Prior completed Europe profile is known before NY opens | Shared resolver | Yes |
| NY_PRIOR_EU_POC | NY RTH | prior Europe | POC | Prior completed Europe profile is known before NY opens | Shared resolver | Yes |
| NY_PRIOR_EU_VAH | NY RTH | prior Europe | VAH | Prior completed Europe profile is known before NY opens | Shared resolver | Yes |
| NY_PRIOR_EU_VAL | NY RTH | prior Europe | VAL | Prior completed Europe profile is known before NY opens | Shared resolver | Yes |
| NY_PRIOR_ASIA_HIGH | NY RTH | target-date Asia | high | Asia closes at 08:00 UTC before NY begins at 13:30 UTC | Shared resolver | Yes |
| NY_PRIOR_ASIA_LOW | NY RTH | target-date Asia | low | Asia closes at 08:00 UTC before NY begins at 13:30 UTC | Shared resolver | Yes |
| NY_PRIOR_ASIA_POC | NY RTH | target-date Asia | POC | Asia closes at 08:00 UTC before NY begins at 13:30 UTC | Shared resolver | Yes |
| NY_PRIOR_ASIA_VAH | NY RTH | target-date Asia | VAH | Asia closes at 08:00 UTC before NY begins at 13:30 UTC | Shared resolver | Yes |
| NY_PRIOR_ASIA_VAL | NY RTH | target-date Asia | VAL | Asia closes at 08:00 UTC before NY begins at 13:30 UTC | Shared resolver | Yes |
| ASIA_PRIOR_NY_HIGH | Asia | prior NY RTH | high | Previous NY completes before 00:00 UTC Asia open | Shared resolver | Yes |
| ASIA_PRIOR_NY_LOW | Asia | prior NY RTH | low | Previous NY completes before 00:00 UTC Asia open | Shared resolver | Yes |
| ASIA_PRIOR_NY_POC | Asia | prior NY RTH | POC | Previous NY completes before 00:00 UTC Asia open | Shared resolver | Yes |
| ASIA_PRIOR_NY_VAH | Asia | prior NY RTH | VAH | Previous NY completes before 00:00 UTC Asia open | Shared resolver | Yes |
| ASIA_PRIOR_NY_VAL | Asia | prior NY RTH | VAL | Previous NY completes before 00:00 UTC Asia open | Shared resolver | Yes |
| ASIA_PRIOR_EU_HIGH | Asia | prior Europe | high | Previous Europe completes before 00:00 UTC Asia open | Shared resolver | Yes |
| ASIA_PRIOR_EU_LOW | Asia | prior Europe | low | Previous Europe completes before 00:00 UTC Asia open | Shared resolver | Yes |
| ASIA_PRIOR_EU_POC | Asia | prior Europe | POC | Previous Europe completes before 00:00 UTC Asia open | Shared resolver | Yes |
| ASIA_PRIOR_EU_VAH | Asia | prior Europe | VAH | Previous Europe completes before 00:00 UTC Asia open | Shared resolver | Yes |
| ASIA_PRIOR_EU_VAL | Asia | prior Europe | VAL | Previous Europe completes before 00:00 UTC Asia open | Shared resolver | Yes |

Every cross-session candidate uses the shared causal level resolver and still
requires its period's causal-tape builder to emit the generic, target-session
interaction population and one shared level catalog. It does not require a
strategy-specific precomputed file.

## CROSS_SESSION_REJECTED_NONCAUSAL

| Would-be IDs | Target | Source | Reason |
|---|---|---|---|
| EU_PRIOR_ASIA_HIGH, EU_PRIOR_ASIA_LOW, EU_PRIOR_ASIA_POC, EU_PRIOR_ASIA_VAH, EU_PRIOR_ASIA_VAL | Europe | target-date Asia | Rejected. Asia’s profile completes at 08:00 UTC, but Europe opens at 07:00 UTC during British Summer Time. It therefore is not known at all Europe target-session signal times. |

## Scale and execution contract

- Same-session candidates: **21**.
- Valid cross-session candidates: **30**.
- Total explicit strategies: **51**.
- Stage 1: `51 × 15 = 765` RR/stop configurations.
- Stage 2: `51 × 38,760 = 1,976,760` five-weight/Q configurations.

These are counts only. No historical replay, data download, or performance
claim was made while creating this universe.
