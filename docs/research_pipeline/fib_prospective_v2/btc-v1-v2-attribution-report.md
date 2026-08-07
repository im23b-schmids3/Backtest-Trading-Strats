# BTC V1/V2 attribution experiment - development only

Candidate `FIB09-BTC-1D-POST0786`; chronology `[2022-01-01T00:00:00Z, 2025-01-01T00:00:00Z)`. This was an in-memory, no-network experiment over sealed Binance USD-M BTCUSDT 1-minute development partitions. No holdout partition was opened, no candidate or parameter changed, and no production artifact or normal run directory was created.

## Control and identity gate

The source contained 1,578,240 development minutes. Its derived HTF population was 1,095 complete UTC daily bars (the terminal daily bar is not a completed, next-bar-executable signal bar). The single frozen V1 proposed setup population was 2,190, with 2,190 unique setup IDs. Its canonical ordered setup-ID serialization was compared byte-for-byte before execution and reused unchanged by B, C, and D. Therefore B=C=D setup identity passed.

A is the owner-attested existing V1 reference: 173 trades, PF 1.637699732419788243039781574, net P&L 31,622.45584540224982560, final equity 41,622.45584540224982560. It uses an owner-attested unknown-source daily file. B uses sealed Binance data; A is comparison-only. A's proposed-setup count, average/median net R, win rate, and max drawdown were not supplied by that attestation and are not derivable from the sealed B/C/D data.

## Funnel and reconciliation

| Variant | execution | proposed / unique setups | orders | filled orders | executed trades | long / short | ACTIVE_POSITION_BLOCKED | ENTRY_EXPIRED | STOP_DISTANCE_REJECTED | SESSION_OR_DATA_END | SESSION_ENTRY_CUTOFF_2245 |
| --- | --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| B | exact frozen V1 daily-bar; overnight allowed; no cutoff | 2,190 / 2,190 | 2,188 | 181 | 181 | 86 / 95 | 1,381 | 626 | 0 | 2 | 0 |
| C | exact B signals; post-activation 1m; overnight allowed; no cutoff | 2,190 / 2,190 | 2,188 | 711 | 711 | 336 / 375 | 1,338 | 71 | 65 | 5 | 0 |
| D | exact B signals/C execution plus 22:45 force-flat and pending cancellation | 2,190 / 2,190 | 2,188 | 691 | 691 | 324 / 367 | 681 | 68 | 45 | 2 | 703 |

All lifecycle reconciliations passed. Equation: executed trades + ACTIVE_POSITION_BLOCKED + ENTRY_EXPIRED + STOP_DISTANCE_REJECTED + SESSION_OR_DATA_END + SESSION_ENTRY_CUTOFF_2245 = proposed setups. B: 181+1,381+626+0+2+0=2,190. C: 711+1,338+71+65+5+0=2,190. D: 691+681+68+45+2+703=2,190. Filled orders = executed trades, all order/setup and trade/order parent links were unique and valid, quantity conservation and gross/net P&L, equity, and event reconciliation all passed.

## Economics and holdings

| Variant | net P&L | final equity | PF | average / median net R | win rate | max DD % | holding minutes: total / median / mean / max | overnight trades | forced exits |
| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| B | 43,385.67612643252030800 | 53,385.67612643252030800 | 1.681638407156786015727906151 | 0.4972126410852612433680702953 / 0.8288375806887225508542464926 | 0.5524861878453038674033149171 | 22.99628456712937079175313745 | 411,840 / 1,440 / 2,275.359116022099447513812155 / 7,200 | 181 | 0 |
| C | -9,986.23172500343118600 | 13.76827499656881400 | 0.4139694706545624661686385179 | -0.4842534896600778771294710662 / -0.1439247693885711305436275447 | 0.4627285513361462728551336146 | 99.86513482949535246410591200 | 41,051 / 19 / 57.73699015471167369901547117 / 2,817 | 7 | 0 |
| D | -9,985.71600563619170400 | 14.28399436380829600 | 0.4021368535916018338452896434 | -0.4906607063489253160498926110 / -0.1439247693885711305436275447 | 0.4587554269175108538350217077 | 99.86008317412003240057341968 | 36,850 / 17 / 53.32850940665701881331403763 / 1,105 | 0 | 10 |

D final exit reasons: 677 STOP, 4 TP5, 10 FORCED_SESSION_EXIT_2245. It had no overnight trade, cutoff entry, position after cutoff, or pending order after cutoff. C's seven overnight trades are expected under its explicitly overnight-permitted rule.

## Deltas and five exact answers

| Transition | setup count | trade count | PF | average net R | net P&L | max DD |
| --- | ---:| ---:| ---:| ---:| ---:| ---:|
| A-to-B | not determinable from A attestation | +8 | +0.043938674736997772688124577 | not determinable from A attestation | +11,763.22028103027048240 | not determinable from A attestation |
| B-to-C | 0 | +530 | -1.267668936502223549559737633 | -0.9814661307453391204971413615 | -53,371.90785143595149400 | +76.86885026236598167 percentage points |
| C-to-D | 0 | -20 | -0.0118326170629606323223488745 | -0.0064072166888474389208215448 | +0.51571936723948200 | -0.00505165537532006 percentage points |

1. Are B/C/D setup IDs identical? Yes, byte-for-byte before execution.
2. Is data-source change the cause of the PF collapse? No: A-to-B improves PF.
3. Is 1-minute execution resolution the largest deterioration? Yes: B-to-C reduces PF by 1.267668936502223549559737633 with no signal change.
4. Is 22:45 force-flat dominant? No: C-to-D changes PF by only -0.0118326170629606323223488745 while removing overnight exposure.
5. Classification is `EXECUTION_RESOLUTION_DOMINANT`; this attributes the largest PF deterioration without tuning.
