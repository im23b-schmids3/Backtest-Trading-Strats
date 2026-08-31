# CME Futures Order Flow Research Framework

A research-oriented framework for studying intraday futures market microstructure, order-flow absorption, liquidity behavior, and execution-aware trading hypotheses using historical CME market data.

The project is designed around **causal event replay, deterministic backtesting, reproducible research artifacts, and explicit separation between signal detection and execution modelling**.

The primary research market is the **E-mini S&P 500 Futures (ES)**, with Micro E-mini S&P 500 Futures (MES) used where appropriate for execution and position-sizing analysis.

> This repository is for quantitative research and historical simulation only.  
> It does not contain the private live-trading infrastructure used for production execution.

---

## Overview

The framework investigates whether short-term price behavior around important structural levels can be explained by observable order-flow and liquidity dynamics.

Research areas include:

- Level 2 market-depth behavior
- Market-by-order / Level 3 order-flow analysis
- Liquidity consumption and restoration
- Bid/ask aggression
- Price resistance after aggressive execution
- Persistence of displayed liquidity
- Multi-level order-book support
- False-refill detection
- Structural price levels
- Current-session high/low sweeps
- Prior-session volume-profile levels
- Intraday market regime diagnostics
- Execution-aware position sizing
- Robustness and parameter-surface analysis

The objective is not simply to maximize historical PnL.

A major focus of the repository is distinguishing between:

- isolated backtest peaks,
- broad parameter plateaus,
- regime-dependent behavior,
- execution-model artifacts,
- and hypotheses that may justify future out-of-sample validation.

---

## Market Data

Historical market data is primarily sourced from **Databento CME Globex datasets**.

Depending on the experiment, the framework can use:

### ES

- Market By Order (`MBO`)
- Market By Price / Level 2 (`MBP`)
- Executed trades

### MES

- Market By Price (`MBP-1`)
- Execution data for native micro-contract simulation

Different datasets are deliberately kept separate because signal generation and execution modelling have different data requirements.

For example:

- ES depth may determine the signal.
- ES trades may construct structural session levels.
- MES market data may be used when ES position sizing exceeds the configured risk budget.

The research pipeline records dataset identity, session windows, source hashes, and other provenance metadata wherever practical.

---

## Research Philosophy

### Causal Replay

All trading decisions are evaluated using information that would have been available at that point in time.

Future observations may be used only after a setup has already qualified, for example to determine:

- confirmation,
- entry,
- stop,
- target,
- session cutoff,
- or retrospective trade outcome.

The framework avoids using completed-session information to make decisions earlier in that same session.

---

### Deterministic Results

Research runs are designed to be reproducible.

Artifacts may include:

- source-population hashes,
- strategy-contract hashes,
- cache hashes,
- configuration manifests,
- trade ledgers,
- interaction ledgers,
- session diagnostics,
- execution-model breakdowns,
- and summary reports.

Where possible, identical inputs and strategy contracts should produce identical outputs.

---

### Fail-Closed Data Handling

Incomplete or ambiguous source data is not silently repaired.

Examples include:

- missing prior-session context,
- incomplete market-data windows,
- unresolved source-end positions,
- insufficient book initialization,
- missing execution data,
- or unverified session coverage.

Such cases are explicitly classified rather than inferred.

---

## Order-Flow Model

A major research branch models local absorption through five normalized components:

### G1 — Aggression

Measures relevant aggressive execution against liquidity near the structural level.

### G2 — Restoration

Measures how liquidity behaves after being consumed, including genuine restoration behavior.

### G3 — Price Resistance

Measures whether aggressive trading fails to produce the expected directional price continuation.

### G4 — Persistence

Measures whether the observed absorption behavior persists rather than appearing as a transient book event.

### G5 — Multi-Level Support

Measures whether supporting liquidity behavior exists across multiple nearby price levels.

A weighted quality score can then be constructed from these components.

Different experiments may test alternative frozen weight configurations and quality thresholds.

Parameter searches are explicitly labelled as **retrospective optimization**, not out-of-sample validation.

---

## Structural Levels

Order-flow interactions can be evaluated around structural market levels such as:

- Prior RTH POC
- Prior RTH High
- Prior RTH Low
- Prior RTH VAH
- Prior RTH VAL
- Current RTH High Sweep
- Current RTH Low Sweep

Additional session-specific research can use equivalent Europe-session structures.

Volume-profile levels are constructed from executed trade data rather than inferred from chart bars.

---

## Interaction Lifecycle

A typical order-flow interaction follows a deterministic lifecycle:

1. Price enters a defined vicinity around a structural level.
2. Relevant executions and depth changes are observed.
3. Liquidity consumption, restoration, persistence, and price response are measured.
4. The interaction either completes or expires.
5. Primitive eligibility rules are evaluated.
6. A weighted quality score is calculated.
7. Qualified interactions may enter a separate confirmation phase.
8. Execution is simulated using the frozen execution model.

Interaction state is intentionally separate from trade state.

This allows the framework to analyze rejected interactions as well as accepted setups.

---

## Confirmation and Execution Research

Signal detection and trade execution are treated as separate research problems.

Experiments include:

- fixed confirmation horizons,
- event-driven favorable-tick confirmation,
- causal execution latency,
- stop/target geometry,
- ES-first position sizing,
- MES fallback,
- session hard-flat behavior,
- and one-position-at-a-time constraints.

Execution assumptions are recorded explicitly rather than hidden inside aggregate backtest results.

---

## Risk Modelling

Research strategies use a nominal risk budget and deterministic contract-sizing logic.

Typical modelling includes:

- fixed dollar risk budget,
- ES-first sizing,
- MES fallback when required,
- contract caps,
- no pyramiding,
- no averaging down,
- predefined stop geometry,
- predefined R-multiple targets.

The purpose is to make results closer to executable futures trading rather than assuming arbitrary fractional position sizes.

---

## Session Research

The framework supports session-specific research rather than assuming a strategy transfers unchanged across all market regimes.

Examples include:

### New York RTH

Used for the primary ES structural-level and absorption research.

### Europe Session

Research can define canonical Europe/London session windows and independently evaluate:

- prior Europe structural levels,
- current Europe high/low sweeps,
- session-specific quality weighting,
- and cross-session transferability.

Session definitions are explicit and timezone-aware.

---

## Parameter Research

Large configuration searches are implemented using compact reusable feature caches whenever possible.

Instead of replaying the complete raw market-data history for every parameter configuration:

1. raw market data is replayed once,
2. causal interaction features are materialized,
3. immutable interaction data is cached,
4. thousands of parameter configurations are evaluated offline.

This dramatically reduces repeated I/O and makes large robustness studies practical.

Examples include:

- Weight × Quality matrices
- Trigger × Target matrices
- robustness neighborhoods
- connected parameter plateaus
- monthly consistency analysis

---

## Robustness Analysis

The repository intentionally avoids treating the highest-PnL configuration as automatically meaningful.

Research reports can evaluate:

- total R
- net PnL
- trade count
- win rate
- profit factor
- average and median R
- maximum drawdown
- monthly performance
- long/short performance
- structural-level performance
- ES vs MES execution
- neighboring parameter configurations
- profitable-neighbor ratios
- connected parameter plateaus
- low-sample warnings

Broad, stable regions of parameter space are generally considered more informative than isolated peaks.

---

## Market Regime Diagnostics

The framework also supports descriptive analysis of the environment surrounding a setup.

Examples include:

- 5-minute momentum
- 15-minute momentum
- 30-minute momentum
- distance from VWAP
- recent trading range
- session range
- execution count
- executed volume
- previous interactions with the same structural level

These diagnostics are used to investigate regime dependence without automatically converting descriptive relationships into trading rules.

---

## Repository Structure

The exact layout evolves with the research, but the main components generally follow this structure:

```text
Trading-Bot-Fib/
│
├── src/
│   └── research_pipeline/
│       └── ...
│
├── tests/
│   └── research_pipeline/
│       └── ...
│
├── research_runs/
│   └── generated research artifacts
│
├── data/
│   └── local historical market data
│
├── docs/
│   └── research and methodology documentation
│
└── README.md
```
Large historical market-data files and local credentials are not intended for version control.

## Research Artifacts

A typical completed experiment may produce:

- summary.json <br>

- run-manifest.json <br>

- frozen-contract.json <br>

- trade-ledger.csv <br>

- interaction-ledger.csv <br>

- configuration-results.csv <br>

- monthly-results.csv <br>

- direction-results.csv <br>

- neighbor-robustness.csv <br>

- plateau-membership.csv <br>

- diagnostic-report.md <br>

The exact files vary by experiment.

Machine-readable artifacts are preferred alongside human-readable reports so results can be independently audited and compared.

## Testing

The project uses automated tests for both strategy semantics and research infrastructure.

Tests cover areas such as:

interaction lifecycle
feature calculations
quality-score mathematics
structural levels
confirmation behavior
stop/target construction
contract sizing
ES/MES fallback
causal timestamp handling
session boundaries
parameter-grid construction
deterministic output
cache integrity
data-leakage prevention
source-end behavior
and research-regression parity


## Reproducibility

Where practical, research runs preserve enough metadata to answer:

- Which data was used? <br>

- Which sessions were included? <br>

- Which sessions were excluded? <br>

- Which strategy contract was used? <br>

- Which execution assumptions were used? <br>

- Which parameter grid was tested? <br>

- Was the analysis retrospective or out-of-sample? <br>

- Did any source-data integrity issue occur? <br>

- Can the same result be reproduced from the same inputs?

This provenance is considered part of the research result itself.

## Out-of-Sample Discipline

The repository distinguishes between several forms of evidence:

development data,
seen historical data,
retrospective robustness testing,
parameter optimization,
diagnostics,
and fresh untouched out-of-sample validation.

A strategy that performs well during retrospective optimization is not considered validated.

Parameter selection should be frozen before future unseen data is evaluated.

## Technology

Core tooling includes:

- Python <br>

- Databento historical CME data <br>

- Smithers for automated workflows <br>

- event-driven market-data replay <br>

- deterministic strategy state machines <br>

- CSV / JSON research artifacts <br>

- SHA-256 provenance and integrity checks <br>

- pytest regression testing <br>

- PowerShell-based research workflows on Windows

The implementation favors explicit deterministic logic over opaque black-box modelling.

## Current Research Direction

The broader research program investigates whether measurable order-flow absorption near important futures market structures contains repeatable information about short-term price behavior.

Particular attention is given to:

distinguishing genuine absorption from simple resting liquidity,
measuring restoration after aggressive consumption,
understanding session-specific behavior,
identifying parameter robustness rather than isolated optimization peaks,
and determining whether retrospective hypotheses survive genuinely fresh data.

## Disclaimer / Important

This repository is provided for research and educational purposes only.

Historical backtests, simulations, parameter studies, and hypothetical performance do not guarantee future results.

Futures trading involves substantial risk and can result in losses greater than expected.

Nothing in this repository constitutes investment advice, a recommendation to trade, or a representation of future performance.