# ranker/data_yahoo.py — Yahoo Finance data fetching (parallel)
import time
import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from ranker.config import CFG, FUNDAMENTAL_FIELDS
from ranker.utils import _safe

def _get_one(ticker: str) -> tuple:
    try:
        t_obj = yf.Ticker(ticker)
        info  = t_obj.info
        data  = {k: info.get(k) for k in FUNDAMENTAL_FIELDS}

        # ── Earnings Revisions (Step 2) ──────────────────────────────
        # eps_revisions: Up/Down analyst count → revision breadth
        # eps_trend:     consensus EPS change over time → revision magnitude
        try:
            rev = t_obj.eps_revisions
            if rev is not None and not rev.empty:
                # Prefer "Current Year" column, fallback to "Next Quarter"
                col = "Current Year" if "Current Year" in rev.columns else (
                      "Next Quarter" if "Next Quarter" in rev.columns else
                      (rev.columns[0] if len(rev.columns) > 0 else None))
                if col is not None:
                    up30   = _safe(rev.at["Up Last 30 Days",   col] if "Up Last 30 Days"   in rev.index else np.nan, 0)
                    down30 = _safe(rev.at["Down Last 30 Days", col] if "Down Last 30 Days" in rev.index else np.nan, 0)
                    up7    = _safe(rev.at["Up Last 7 Days",    col] if "Up Last 7 Days"    in rev.index else np.nan, 0)
                    down7  = _safe(rev.at["Down Last 7 Days",  col] if "Down Last 7 Days"  in rev.index else np.nan, 0)
                    total30 = up30 + down30
                    total7  = up7  + down7
                    data["rev_up30"]    = up30
                    data["rev_down30"]  = down30
                    data["rev_up7"]     = up7
                    data["rev_down7"]   = down7
                    # Revision ratio: +1 = all up, -1 = all down, 0 = balanced
                    data["rev_ratio_30d"] = (up30 - down30) / total30 if total30 > 0 else np.nan
                    data["rev_ratio_7d"]  = (up7  - down7)  / total7  if total7  > 0 else np.nan
        except Exception:
            pass  # graceful — no revision data for this ticker

        try:
            trend = t_obj.eps_trend
            if trend is not None and not trend.empty:
                col = "Current Year" if "Current Year" in trend.columns else (
                      "Next Quarter" if "Next Quarter" in trend.columns else
                      (trend.columns[0] if len(trend.columns) > 0 else None))
                if col is not None:
                    current = _safe(trend.at["Current Estimate", col] if "Current Estimate" in trend.index else np.nan)
                    ago_90  = _safe(trend.at["90 Days Ago",      col] if "90 Days Ago"      in trend.index else np.nan)
                    ago_30  = _safe(trend.at["30 Days Ago",      col] if "30 Days Ago"      in trend.index else np.nan)
                    data["eps_est_current"] = current
                    data["eps_est_90d_ago"] = ago_90
                    data["eps_est_30d_ago"] = ago_30
                    # % change in consensus EPS over 90 days
                    if not np.isnan(current) and not np.isnan(ago_90) and abs(ago_90) > 0.01:
                        data["eps_revision_pct_90d"] = (current - ago_90) / abs(ago_90)
                    if not np.isnan(current) and not np.isnan(ago_30) and abs(ago_30) > 0.01:
                        data["eps_revision_pct_30d"] = (current - ago_30) / abs(ago_30)
        except Exception:
            pass  # graceful — no trend data for this ticker

        return ticker, data
    except Exception:
        return ticker, {}


# ── Diagnostic probe: test eps_revisions on a few tickers before bulk fetch ──
def _probe_earnings_revisions(sample_tickers: list):
    """Run once before parallel fetch to diagnose eps_revisions availability."""
    print("  🔍 Probing eps_revisions/eps_trend on sample tickers...")
    for tk in sample_tickers[:3]:
        try:
            t_obj = yf.Ticker(tk)
            rev = t_obj.eps_revisions
            trend = t_obj.eps_trend
            rev_status = "None" if rev is None else ("empty" if rev.empty else f"OK {rev.shape} cols={list(rev.columns)} idx={list(rev.index)}")
            trend_status = "None" if trend is None else ("empty" if trend.empty else f"OK {trend.shape} cols={list(trend.columns)} idx={list(trend.index)}")
            print(f"    {tk}: eps_revisions={rev_status}")
            print(f"    {tk}: eps_trend={trend_status}")
        except Exception as e:
            print(f"    {tk}: ERROR — {e}")
    print()


def fetch_yf_parallel(tickers: list) -> dict:
    results = {}
    with ThreadPoolExecutor(max_workers=CFG["max_workers_yf"]) as executor:
        futures = {executor.submit(_get_one, t): t for t in tickers}
        for future in tqdm(as_completed(futures), total=len(tickers),
                           desc="Yahoo Finance (parallel)"):
            t, info = future.result()
            results[t] = info
    return results   # always a dict, never None


def fetch_price_multi(tickers: list) -> pd.DataFrame:
    try:
        raw = yf.download(tickers, period="2y", auto_adjust=True,
                          progress=False, threads=True)["Close"]
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(tickers[0])
        return raw
    except Exception:
        return pd.DataFrame()


def fetch_spy_returns() -> dict:
    """Fetch SPY 2y price and return 12m/6m/3m/1m returns (decimal)."""
    time.sleep(2.0)  # FIX: avoid rate-limit after bulk stock downloads
    for attempt in range(2):
        try:
            raw = yf.download("SPY", period="2y", auto_adjust=True, progress=False, threads=False)
            if raw.empty:
                continue
            close = None
            if hasattr(raw.columns, "get_level_values") and isinstance(raw.columns, pd.MultiIndex):
                try:
                    close = raw["Close"].squeeze()
                except (KeyError, TypeError):
                    for col in raw.columns:
                        if (isinstance(col, tuple) and "Close" in col) or col == "Close":
                            close = raw[col].squeeze()
                            break
            elif "Close" in raw.columns:
                close = raw["Close"].squeeze()
            if close is None:
                continue
            s = pd.Series(close).dropna()
            if len(s) < 21:
                continue
            out = {}
            for n_days, key in [(252, "12m"), (126, "6m"), (63, "3m"), (21, "1m")]:
                out[key] = (float(s.iloc[-1] / s.iloc[-n_days] - 1)) if len(s) >= n_days else np.nan
            print(f"  ✅  SPY returns: 12m={out.get('12m', np.nan):.2%} 6m={out.get('6m', np.nan):.2%}")
            return out
        except Exception as e:
            if attempt == 0:
                time.sleep(5.0)  # FIX: longer retry delay for SPY
            else:
                print(f"  ⚠️  SPY fetch failed: {e} — RS columns will be NaN")
    return {}


def add_price_momentum(df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    prices = fetch_price_multi(tickers)
    if prices.empty:
        for col in ["perf_12m", "perf_6m", "perf_3m", "perf_1m", "momentum_composite",
                    "rs_12m", "rs_6m", "rs_3m", "rs_1m", "rs_composite",
                    "ma_regime_score", "pct_above_ma200"]:
            df[col] = np.nan
        return df

    def perf_window(n_days: int) -> dict:
        out = {}
        for col in prices.columns:
            s = prices[col].dropna()
            out[col] = s.iloc[-1] / s.iloc[-n_days] - 1 if len(s) >= n_days else np.nan
        return out

    p12, p6, p3, p1 = perf_window(252), perf_window(126), perf_window(63), perf_window(21)
    df["perf_12m"] = df["ticker"].map(p12)
    df["perf_6m"]  = df["ticker"].map(p6)
    df["perf_3m"]  = df["ticker"].map(p3)
    df["perf_1m"]  = df["ticker"].map(p1)
    df["momentum_composite"] = (
        0.50 * df["perf_12m"].fillna(0) + 0.30 * df["perf_6m"].fillna(0) +
        0.15 * df["perf_3m"].fillna(0)  + 0.05 * df["perf_1m"].fillna(0)
    )
    all_nan = df[["perf_12m", "perf_6m", "perf_3m", "perf_1m"]].isna().all(axis=1)
    df.loc[all_nan, "momentum_composite"] = np.nan

    spy = fetch_spy_returns()
    spy_12m = spy.get("12m", np.nan)
    spy_6m  = spy.get("6m",  np.nan)
    spy_3m  = spy.get("3m",  np.nan)
    spy_1m  = spy.get("1m",  np.nan)
    df["rs_12m"] = df["perf_12m"] - spy_12m
    df["rs_6m"]  = df["perf_6m"]  - spy_6m
    df["rs_3m"]  = df["perf_3m"]  - spy_3m
    df["rs_1m"]  = df["perf_1m"]  - spy_1m
    df["rs_composite"] = (
        0.50 * df["rs_12m"].fillna(0) + 0.30 * df["rs_6m"].fillna(0) +
        0.15 * df["rs_3m"].fillna(0)  + 0.05 * df["rs_1m"].fillna(0)
    )
    rs_all_nan = df[["rs_12m", "rs_6m", "rs_3m", "rs_1m"]].isna().all(axis=1)
    df.loc[rs_all_nan, "rs_composite"] = np.nan

    for col in ["perf_12m", "perf_6m", "perf_3m", "perf_1m", "momentum_composite"]:
        df[col] = df[col].clip(-0.80, 5.0)
    for col in ["rs_12m", "rs_6m", "rs_3m", "rs_1m", "rs_composite"]:
        df[col] = df[col].clip(-0.80, 5.0)

    # ── MA Regime Signal ─────────────────────────────────────────
    # Sweet-spot scoring: rewards stocks just above rising MA200
    # while penalizing over-extended stocks (>25% above).
    # This captures the "on sale in an uptrend" thesis:
    #   - Just crossed above rising MA200 = best entry (score 2.0)
    #   - Moderately above = still good (1.5)
    #   - Extended = reduced benefit (0.5)
    #   - Over-extended or below declining MA = no benefit (0.0)
    #
    # References: Brock/Lakonishok/LeBaron 1992, Minervini Stage Analysis
    def _ma_regime(prices_df: pd.DataFrame) -> dict:
        regime = {}
        for col in prices_df.columns:
            try:
                s = prices_df[col].dropna()
                if len(s) < 200:
                    regime[col] = np.nan
                    continue
                price    = float(s.iloc[-1])
                ma200    = float(s.iloc[-200:].mean())
                ma150    = float(s.iloc[-150:].mean()) if len(s) >= 150 else ma200
                # MA direction: compare current MA200 to MA200 from 20 days ago
                ma200_20d_ago = float(s.iloc[-220:-20].mean()) if len(s) >= 220 else ma200
                ma_rising = ma200 > ma200_20d_ago

                if ma200 <= 0:
                    regime[col] = np.nan
                    continue

                pct_above = (price / ma200) - 1.0  # e.g. 0.05 = 5% above

                if pct_above < 0:
                    # Below MA200
                    regime[col] = 0.0
                elif not ma_rising:
                    # Above MA200 but MA is flat/declining — weak signal
                    regime[col] = 0.5 if pct_above <= 0.15 else 0.0
                else:
                    # Above rising MA200 — the sweet spot zone
                    if pct_above <= 0.05:
                        regime[col] = 2.0    # just crossed — best entry
                    elif pct_above <= 0.15:
                        regime[col] = 1.5    # healthy uptrend
                    elif pct_above <= 0.25:
                        regime[col] = 0.5    # getting extended
                    else:
                        regime[col] = 0.0    # over-extended — no bonus

                # Bonus: above BOTH MA150 and MA200, both rising → confirmed Stage 2
                if pct_above > 0 and ma_rising and price > ma150:
                    ma150_20d_ago = float(s.iloc[-170:-20].mean()) if len(s) >= 170 else ma150
                    if ma150 > ma150_20d_ago:
                        regime[col] = min(regime[col] + 0.5, 2.5)  # Stage 2 bonus, capped

            except Exception:
                regime[col] = np.nan
        return regime

    ma_regime = _ma_regime(prices)
    df["ma_regime_score"] = df["ticker"].map(ma_regime)
    # Also store pct above MA200 for display/debugging
    def _pct_above_ma200(prices_df):
        out = {}
        for col in prices_df.columns:
            try:
                s = prices_df[col].dropna()
                if len(s) >= 200:
                    out[col] = round(float(s.iloc[-1]) / float(s.iloc[-200:].mean()) - 1.0, 4)
                else:
                    out[col] = np.nan
            except Exception:
                out[col] = np.nan
        return out
    df["pct_above_ma200"] = df["ticker"].map(_pct_above_ma200(prices))

    return df
