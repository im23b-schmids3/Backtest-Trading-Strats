from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from fib_backtester.backtest.engine import BacktestEngine
from fib_backtester.config import AssetConfig, RunConfig
from fib_backtester.data.cache import Cache

FEATURE_COLUMNS = ["ema50_gap", "ema100_gap", "ema200_gap", "ema200_slope", "atr_pct", "realized_vol", "rsi14", "roc10", "bb_width", "volume_ratio", "timeframe_hours", "side_long", "btc_ema200_gap"]


def run_research(config: RunConfig, root: str | Path = "reports/research") -> dict:
    """Run fixed-search walk-forward selection and a final holdout evaluation once."""
    np.random.seed(config.seed)
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    pairs = _available_pairs(config)
    if not pairs:
        raise RuntimeError("no validated cached series available for research")
    pd.DataFrame([{"asset": asset, "timeframe": timeframe, "actual_start": bars.index[0], "actual_end": bars.index[-1], "candles": len(bars)} for asset, timeframe, bars in pairs]).to_csv(root / "period_coverage.csv", index=False)
    trial_rows, folds = [], []
    selected = []
    for asset, timeframe, bars in pairs:
        boundaries = _boundaries(bars.index)
        candidate_scores = []
        for n in range(2, 11):
            for distance in (5, 10, 15):
                scores = []
                for fold_no, (train_start, train_end, validation_start, validation_end) in enumerate(boundaries["folds"], 1):
                    trades = _candidate_trades(config, asset, timeframe, bars, n, distance, validation_start, validation_end)
                    stats = _trade_stats(trades, config.initial_cash)
                    score = _robust_score(stats)
                    scores.append(score)
                    folds.append({"asset": asset, "timeframe": timeframe, "swing_n": n, "min_pivot_distance": distance, "fold": fold_no,
                                  "train_start": train_start, "train_end": train_end, "validation_start": validation_start, "validation_end": validation_end,
                                  "score": score, **stats})
                stability = float(np.std(scores))
                aggregate = float(np.mean(scores) - stability)
                row = {"asset": asset, "timeframe": timeframe, "swing_n": n, "min_pivot_distance": distance,
                       "mean_validation_score": float(np.mean(scores)), "fold_score_std": stability, "selection_score": aggregate,
                       "trial_count": len(scores), "holdout_start": boundaries["holdout_start"]}
                trial_rows.append(row); candidate_scores.append(row)
        selected.append(max(candidate_scores, key=lambda row: row["selection_score"]))
    trials = pd.DataFrame(trial_rows); fold_frame = pd.DataFrame(folds); selected_frame = pd.DataFrame(selected)
    _write_sqlite(root / "optimization.sqlite", trials, fold_frame, selected_frame)
    trials.to_csv(root / "trial_history.csv", index=False); fold_frame.to_csv(root / "walk_forward_folds.csv", index=False); selected_frame.to_csv(root / "selected_configurations.csv", index=False)
    holdout_trades, training_features, holdout_features, holdout_rows = _holdout_and_features(config, pairs, selected_frame)
    holdout_rows.to_csv(root / "final_holdout_results.csv", index=False)
    training_features.to_csv(root / "training_trade_features.csv", index=False); holdout_features.to_csv(root / "holdout_trade_features.csv", index=False)
    regime = _regime_tables(training_features, holdout_features); regime.to_csv(root / "regime_performance.csv", index=False)
    direction = _direction_table(training_features, holdout_features); direction.to_csv(root / "direction_performance.csv", index=False)
    importance, ml_results = _ml_filter(training_features, holdout_features, config.seed)
    importance.to_csv(root / "feature_importance.csv", index=False); ml_results.to_csv(root / "ml_filter_results.csv", index=False)
    robustness = _robustness(config, pairs, selected_frame); robustness.to_csv(root / "robustness_matrix.csv", index=False)
    monte = _monte_carlo(holdout_features, config.seed); monte.to_csv(root / "monte_carlo.csv", index=False)
    warning = pd.DataFrame([{"parameter_trials": len(trials), "final_holdout_trades": len(holdout_features), "warning": "Selection bias risk is high: 243 parameter trials were evaluated; no statistical-significance claim is justified."}])
    warning.to_csv(root / "multiple_testing_warning.csv", index=False)
    _write_report(root, selected_frame, holdout_rows, ml_results, regime, robustness, monte)
    return {"pairs": len(pairs), "trials": len(trials), "holdout_rows": len(holdout_rows), "feature_rows": len(training_features), "root": str(root)}


def _available_pairs(config: RunConfig) -> list[tuple[str, str, pd.DataFrame]]:
    result = []
    for asset in config.assets:
        for timeframe in config.timeframes:
            try:
                bars = Cache().read(asset, timeframe, config.asset_configs[asset].source == "yfinance")
                result.append((asset, timeframe, bars))
            except (FileNotFoundError, ValueError):
                continue
    return result


def _boundaries(index: pd.DatetimeIndex) -> dict:
    start, end = index[0], index[-1]
    holdout_start = end - pd.Timedelta(days=365)
    pre = holdout_start - start
    train_end_1 = start + pre * .45
    validation_start_1, validation_end_1 = train_end_1, start + pre * .65
    train_end_2 = validation_end_1
    return {"holdout_start": holdout_start, "folds": [(start, train_end_1, validation_start_1, validation_end_1), (start, train_end_2, train_end_2, holdout_start)]}


def _candidate_trades(config: RunConfig, asset: str, timeframe: str, bars: pd.DataFrame, n: int, distance: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    position = bars.index.searchsorted(start)
    warmup = bars.iloc[max(0, position - 250):bars.index.searchsorted(end, side="right")]
    run = replace(config, assets=[asset], timeframes=[timeframe], swing_n=n, min_pivot_distance=distance, max_positions=1)
    trades, _ = BacktestEngine(run).run({asset: warmup})
    if trades.empty:
        return trades
    fills = pd.to_datetime(trades["fill_timestamp"], utc=True)
    return trades.loc[(fills >= start) & (fills <= end)].copy()


def _trade_stats(trades: pd.DataFrame, capital: float) -> dict:
    if trades.empty:
        return {"return": 0.0, "trade_count": 0, "profit_factor": 0.0, "max_drawdown": 0.0, "median_r": 0.0}
    pnl = trades.net_pnl.astype(float)
    curve = capital + pnl.cumsum(); drawdown = curve / curve.cummax() - 1
    wins, losses = pnl[pnl > 0].sum(), pnl[pnl < 0].sum()
    return {"return": float(pnl.sum() / capital), "trade_count": int(len(trades)), "profit_factor": float(wins / abs(losses)) if losses else 0.0,
            "max_drawdown": float(drawdown.min()), "median_r": float((pnl / trades.risk_budget).median())}


def _robust_score(stats: dict) -> float:
    return stats["return"] - .5 * abs(stats["max_drawdown"]) + .1 * min(stats["median_r"], 2) + .05 * min(stats["profit_factor"], 3) - .5 * max(0, 20 - stats["trade_count"]) / 20


def _holdout_and_features(config: RunConfig, pairs: list[tuple[str, str, pd.DataFrame]], selected: pd.DataFrame):
    train_features, hold_features, rows, hold_trades = [], [], [], []
    lookup = {(asset, timeframe): bars for asset, timeframe, bars in pairs}
    for choice in selected.to_dict("records"):
        asset, timeframe = choice["asset"], choice["timeframe"]; bars = lookup[(asset, timeframe)]
        holdout_start = pd.Timestamp(choice["holdout_start"])
        all_trades = _candidate_trades(config, asset, timeframe, bars, int(choice["swing_n"]), int(choice["min_pivot_distance"]), bars.index[0], bars.index[-1])
        if all_trades.empty:
            rows.append({**choice, "holdout_trade_count": 0, "holdout_return": 0.0}); continue
        features = trade_features(all_trades, bars, asset, timeframe, lookup.get(("BTC", timeframe)))
        fills = pd.to_datetime(features.fill_timestamp, utc=True)
        training, holdout = features.loc[fills < holdout_start], features.loc[fills >= holdout_start]
        train_features.append(training); hold_features.append(holdout); hold_trades.append(holdout)
        stats = _trade_stats(holdout, config.initial_cash)
        rows.append({**choice, "holdout_start": holdout_start, "holdout_trade_count": len(holdout), **{f"holdout_{k}": v for k, v in stats.items()}})
    return hold_trades, pd.concat(train_features, ignore_index=True) if train_features else pd.DataFrame(), pd.concat(hold_features, ignore_index=True) if hold_features else pd.DataFrame(), pd.DataFrame(rows)


def trade_features(trades: pd.DataFrame, bars: pd.DataFrame, asset: str, timeframe: str, btc_bars: pd.DataFrame | None = None) -> pd.DataFrame:
    """Decision-time feature rows: every indicator is shifted one completed candle."""
    features = _bar_features(bars)
    btc = _bar_features(btc_bars)["ema200_gap"] if btc_bars is not None and asset != "BTC" else None
    rows = []
    for trade in trades.to_dict("records"):
        entry = pd.Timestamp(trade["fill_timestamp"])
        previous = features.loc[features.index < entry]
        if previous.empty: continue
        row = previous.iloc[-1].to_dict()
        if btc is not None:
            prior_btc = btc.loc[btc.index < entry]; row["btc_ema200_gap"] = float(prior_btc.iloc[-1]) if not prior_btc.empty else np.nan
        else: row["btc_ema200_gap"] = np.nan
        row.update(trade); row["asset"] = asset; row["timeframe"] = timeframe; row["timeframe_hours"] = {"1h": 1, "4h": 4, "1d": 24}[timeframe]
        row["side_long"] = float(trade["side"] == "long"); row["tp1_before_stop"] = int(trade["targets_hit"] >= 1); row["profitable"] = int(trade["net_pnl"] > 0)
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.dropna(subset=["ema200_gap", "atr_pct", "rsi14", "net_pnl"]).reset_index(drop=True) if not frame.empty else frame


def _bar_features(bars: pd.DataFrame) -> pd.DataFrame:
    close, high, low, volume = bars.close.astype(float), bars.high.astype(float), bars.low.astype(float), bars.volume.astype(float)
    ema50, ema100, ema200 = close.ewm(span=50, adjust=False).mean(), close.ewm(span=100, adjust=False).mean(), close.ewm(span=200, adjust=False).mean()
    true_range = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    delta = close.diff(); gains, losses = delta.clip(lower=0).rolling(14).mean(), (-delta.clip(upper=0)).rolling(14).mean()
    std = close.rolling(20).std(); mid = close.rolling(20).mean()
    regime = np.where((close > ema200) & (ema200.diff(10) > 0), "bull", np.where((close < ema200) & (ema200.diff(10) < 0), "bear", "sideways"))
    result = pd.DataFrame({"ema50_gap": close/ema50-1, "ema100_gap": close/ema100-1, "ema200_gap": close/ema200-1,
        "ema200_slope": ema200.pct_change(10), "atr_pct": true_range.rolling(14).mean()/close, "realized_vol": close.pct_change().rolling(20).std(),
        "rsi14": 100 - 100/(1 + gains/losses), "roc10": close.pct_change(10), "bb_width": 4*std/mid,
        "volume_ratio": volume/volume.rolling(20).mean(), "regime": regime}, index=bars.index)
    return result.shift(1)


def _ml_filter(train: pd.DataFrame, holdout: pd.DataFrame, seed: int):
    if len(train) < 40 or train.tp1_before_stop.nunique() < 2 or holdout.empty:
        return pd.DataFrame(columns=["feature", "importance"]), pd.DataFrame([{"status": "skipped_insufficient_chronological_samples"}])
    train = train.sort_values("fill_timestamp"); cut = int(len(train)*.7); x_train, x_val = train[FEATURE_COLUMNS].fillna(0), train.iloc[cut:][FEATURE_COLUMNS].fillna(0)
    y_train, y_val = train.tp1_before_stop, train.iloc[cut:].tp1_before_stop
    models = {"logistic": make_pipeline(StandardScaler(), LogisticRegression(C=.2, max_iter=2000, random_state=seed)), "tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=seed)}
    results, fitted = [], {}
    for name, model in models.items():
        model.fit(x_train.iloc[:cut], y_train.iloc[:cut]); prob = model.predict_proba(x_val)[:, 1]
        auc = roc_auc_score(y_val, prob) if y_val.nunique() > 1 else .5
        results.append({"model": name, "validation_auc": auc}); fitted[name] = model
    best = max(results, key=lambda row: row["validation_auc"])["model"]; model = models[best].fit(x_train, y_train)
    prob = model.predict_proba(holdout[FEATURE_COLUMNS].fillna(0))[:, 1]; selected = holdout.loc[prob >= .55]
    results.append({"model": best, "status": "final_holdout", "threshold": .55, "holdout_trades": len(holdout), "selected_trades": len(selected), "holdout_net_pnl": float(holdout.net_pnl.sum()), "filtered_net_pnl": float(selected.net_pnl.sum())})
    perm = permutation_importance(model, x_val, y_val, n_repeats=20, random_state=seed, scoring="roc_auc") if y_val.nunique() > 1 else None
    importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance", ascending=False) if perm else pd.DataFrame(columns=["feature", "importance"])
    return importance, pd.DataFrame(results)


def _regime_tables(train: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    data = pd.concat([train.assign(sample="pre_holdout"), holdout.assign(sample="holdout")], ignore_index=True)
    if data.empty: return pd.DataFrame()
    rule = data.groupby(["sample", "regime"], dropna=False).agg(trades=("net_pnl", "size"), net_pnl=("net_pnl", "sum"), win_rate=("profitable", "mean"), tp1_rate=("tp1_before_stop", "mean")).reset_index().rename(columns={"regime": "label"})
    rule["method"] = "rule_based"
    usable = data[FEATURE_COLUMNS].fillna(0)
    if len(data) >= 12:
        labels = KMeans(n_clusters=3, random_state=42, n_init=20).fit_predict(StandardScaler().fit_transform(usable))
        clustered = data.assign(label=[f"cluster_{value}" for value in labels]).groupby(["sample", "label"], dropna=False).agg(trades=("net_pnl", "size"), net_pnl=("net_pnl", "sum"), win_rate=("profitable", "mean"), tp1_rate=("tp1_before_stop", "mean")).reset_index()
        clustered["method"] = "kmeans_3"
        return pd.concat([rule, clustered], ignore_index=True)
    return rule


def _direction_table(train: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    data = pd.concat([train.assign(sample="pre_holdout"), holdout.assign(sample="holdout")], ignore_index=True)
    if data.empty: return pd.DataFrame()
    return data.groupby(["sample", "side"], dropna=False).agg(trades=("net_pnl", "size"), net_pnl=("net_pnl", "sum"), win_rate=("profitable", "mean"), average_holding_hours=("holding_hours", "mean")).reset_index()


def _robustness(config: RunConfig, pairs, selected: pd.DataFrame) -> pd.DataFrame:
    lookup = {(a,t): b for a,t,b in pairs}; rows=[]
    for choice in selected.to_dict("records"):
        asset, timeframe = choice["asset"], choice["timeframe"]; bars=lookup[(asset,timeframe)]; start=pd.Timestamp(choice["holdout_start"])
        base = _candidate_trades(config, asset,timeframe,bars,int(choice["swing_n"]),int(choice["min_pivot_distance"]),start,bars.index[-1])
        rows.append({"asset":asset,"timeframe":timeframe,"scenario":"conservative_base","net_pnl":float(base.net_pnl.sum()) if not base.empty else 0,"trades":len(base)})
        for label, policy, multiplier in (("optimistic", "optimistic", 1), ("double_cost", "conservative", 2)):
            ac=config.asset_configs[asset]; costs=replace(ac, fee_rate=ac.fee_rate*multiplier, slippage_rate=ac.slippage_rate*multiplier)
            rc=replace(config, asset_configs={**config.asset_configs, asset:costs}, execution_policy=policy)
            trades=_candidate_trades(rc,asset,timeframe,bars,int(choice["swing_n"]),int(choice["min_pivot_distance"]),start,bars.index[-1])
            rows.append({"asset":asset,"timeframe":timeframe,"scenario":label,"net_pnl":float(trades.net_pnl.sum()) if not trades.empty else 0,"trades":len(trades)})
    return pd.DataFrame(rows)


def _monte_carlo(holdout: pd.DataFrame, seed: int) -> pd.DataFrame:
    if holdout.empty: return pd.DataFrame()
    pnl=holdout.net_pnl.to_numpy(float); rng=np.random.default_rng(seed); values=[]
    for _ in range(1000):
        curve=np.cumsum(rng.permutation(pnl)); dd=(curve-np.maximum.accumulate(np.r_[0,curve])[1:]).min(); values.append(dd)
    mean_ci=bootstrap((pnl,), np.mean, n_resamples=2000, random_state=rng).confidence_interval
    return pd.DataFrame([{"trials":1000,"median_reordered_max_drawdown":float(np.median(values)),"p05_reordered_max_drawdown":float(np.quantile(values,.05)),"mean_trade_pnl":float(pnl.mean()),"mean_trade_pnl_ci_low":float(mean_ci.low),"mean_trade_pnl_ci_high":float(mean_ci.high)}])


def _write_sqlite(path: Path, trials: pd.DataFrame, folds: pd.DataFrame, selected: pd.DataFrame) -> None:
    with sqlite3.connect(path) as con:
        trials.to_sql("trials",con,if_exists="replace",index=False); folds.to_sql("folds",con,if_exists="replace",index=False); selected.to_sql("selected",con,if_exists="replace",index=False)


def _write_report(root: Path, selected, holdout, ml, regime, robustness, monte) -> None:
    sections=[]
    for title, table in (("Validation-selected configurations",selected),("Final untouched holdout",holdout),("ML filter",ml),("Regime performance",regime),("Robustness",robustness),("Monte Carlo",monte)):
        sections.append(f"<h2>{title}</h2>{table.to_html(index=False) if not table.empty else '<p>Insufficient data.</p>'}")
    (root/"research_report.html").write_text("<html><body><h1>Chronological Fibonacci research</h1><p>Final holdout was excluded from parameter/model selection. Results are historical and subject to multiple-testing risk.</p>"+"".join(sections)+"</body></html>",encoding="utf-8")
