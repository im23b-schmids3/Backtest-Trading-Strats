# Fibonacci Retracement Backtester

An event-driven, reproducible research baseline for BTC, ETH, SOL, XRP, and the **GC=F gold-futures proxy**. It detects swings only after `N` future candles have closed, submits a 0.882-Fib resting order on the next candle, sizes initial-stop risk at 2% of realized equity, and supports five partial take-profits.

## Setup

Python 3.12+ is required (the current workspace uses Python 3.13). Create and activate a virtual environment, then install the locked project requirements represented by `pyproject.toml`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## Commands

```powershell
# Cache completed public candles (use --refresh to redownload)
python -m fib_backtester.cli download --assets BTC ETH SOL XRP GOLD --timeframes 1h 4h 1d

# Verify UTC ordering, duplicates, price ranges, and expected intervals
python -m fib_backtester.cli validate --config config/default.yaml

# Run every configured asset as one portfolio for each configured timeframe
python -m fib_backtester.cli backtest --config config/default.yaml

# Run N=2..10 and distances 5, 10, 15 against cached data
python -m fib_backtester.cli grid --config config/default.yaml

# Chronological walk-forward / regime / ML-filter research (offline cached data)
python -m fib_backtester.cli research --config config/default.yaml

pytest
```

Each run writes `trades.csv`, `equity.csv`, `monthly_returns.csv`, `metrics.json`, `config.json`, and a self-contained `report.html` under `reports/runs/`. Run summaries go to `reports/summary_grid.csv`.

## Strategy and safeguards

Fib conventions, partial percentages, fees, slippage, setup invalidation, and execution policies are implemented directly in the package and tested. The default intrabar policy is conservative: when a stop and target are both reachable from the same OHLC candle, the adverse stop happens first. For `4h`/`1d`, `lower_timeframe_replay` replays cached `1h` bars; it fails if those bars are unavailable and is never silently approximated.

Default crypto source is Binance spot through CCXT. The default gold source is Yahoo Finance `GC=F`, which is explicitly a COMEX futures proxy. Yahoo’s hourly history and its absence of a reliable `4h` interval are known limitations; commands fail plainly rather than substitute data. See [assumptions](docs/assumptions.md) and the [implementation plan](docs/implementation_plan.md).
