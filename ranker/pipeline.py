# ranker/pipeline.py — Main orchestration
import os, time
import numpy as np
import pandas as pd
from ranker.config import CFG, _TR_AVAILABLE
from ranker import cache

def run_pipeline(use_cache: bool = True) -> pd.DataFrame:
    global _SECTOR_THRESHOLDS, _TR_AVAILABLE

    print("=" * 65)
    print(f"  S&P 500 ADVANCED RANKING v5.3 – GitHub Edition")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    # 1. Cache check
    cached = load_cache() if use_cache else None
    if cached is not None:
        df = cached
        print("  Skipping fetch – using cached data")
    else:
        # 2. Universe
        universe = get_sp500_tickers()
        tickers  = universe["ticker"].tolist()

        # 3. Yahoo Finance (parallel)
        print(f"\n[1/6]  Yahoo Finance ({len(tickers)} tickers, parallel)...")
        _probe_earnings_revisions(tickers)  # diagnostic: test 3 tickers before bulk
        yf_data = fetch_yf_parallel(tickers)   # always returns dict, never None
        fund_df = pd.DataFrame.from_dict(yf_data, orient="index")
        fund_df.index.name = "ticker"
        fund_df = fund_df.reset_index()
        # ── Earnings revision coverage stats ──
        _rev_keys = ["rev_ratio_30d", "eps_revision_pct_90d"]
        for _rk in _rev_keys:
            if _rk in fund_df.columns:
                _n = fund_df[_rk].notna().sum()
                print(f"  📊 {_rk}: {_n}/{len(fund_df)} stocks have data ({_n*100//max(len(fund_df),1)}%)")
            else:
                print(f"  ⚠️  {_rk}: column not in data — eps_revisions/eps_trend may not be available in this yfinance version")
        df = universe.merge(fund_df, on="ticker", how="left")

        # 4. TipRanks
        print("\n[2/6]  TipRanks...")
        tr_df = fetch_tipranks(tickers)
        if not tr_df.empty and "ticker" in tr_df.columns:
            df = df.merge(tr_df, on="ticker", how="left")
            _tr_coverage = tr_df["tr_smart_score"].notna().sum() / max(len(tickers), 1)
            if _tr_coverage < 0.10:
                _TR_AVAILABLE = False
                print(f"  ⚠️  TR coverage only {_tr_coverage:.0%} — switching to fallback weights")
            else:
                _TR_AVAILABLE = True
        else:
            _TR_AVAILABLE = False
            print("  ⚠️  TipRanks unavailable — using fundamental-only weights")
            _TR_COLS = list(_parse_tipranks({}).keys())
            for col in _TR_COLS:
                df[col] = np.nan

        # 5. Computed metrics
        print("\n[3/6]  Computing metrics...")
        df["piotroski_score"]        = df.apply(compute_piotroski,        axis=1)
        df["altman_z"]               = df.apply(compute_altman,           axis=1)
        # PEG fallback: pegRatio from Yahoo is often missing → compute from PE / epsGrowth
        def _compute_peg(row):
            peg = row.get("pegRatio")
            if peg is not None and not (isinstance(peg, float) and np.isnan(peg)):
                return peg
            pe = row.get("trailingPE")
            eg = row.get("earningsGrowth")  # fraction e.g. 0.15 = 15%
            if pe is not None and eg is not None and eg > 0.01:
                try:
                    return round(float(pe) / (float(eg) * 100), 2)
                except Exception:
                    return np.nan
            return np.nan
        df["pegRatio"] = df.apply(_compute_peg, axis=1)
        df["roic"]                   = df.apply(compute_roic,             axis=1)
        fcf_cols = df.apply(compute_fcf_metrics, axis=1, result_type="expand")
        df = pd.concat([df, fcf_cols], axis=1)
        df["pt_upside"]              = df.apply(compute_pt_upside,        axis=1)
        df["tr_pt_upside"]           = df.apply(compute_tr_pt_upside,     axis=1)
        df["earnings_quality_score"] = df.apply(compute_earnings_quality, axis=1)
        df["earnings_revision_score"] = df.apply(compute_earnings_revision_score, axis=1)

        # FIX v5.3: Insider buying as % of market cap (stronger signal than directional binary)
        df["insider_pct_mcap"] = df.apply(
            lambda r: _safe(r.get("tr_insider_3m_usd")) / _safe(r.get("marketCap"), 1)
            if _safe(r.get("tr_insider_3m_usd")) and _safe(r.get("marketCap")) > 0 else np.nan,
            axis=1
        )

        # FIX v5.3: Ensure eps_revision_pct_30d exists (may not if eps_trend unavailable)
        if "eps_revision_pct_30d" not in df.columns:
            df["eps_revision_pct_30d"] = np.nan

        # Clip financial outliers (negative equity → crazy ROE, tiny mktcap → crazy FCF yield)
        df["returnOnEquity"] = df["returnOnEquity"].clip(-2.0, 5.0)
        df["fcf_yield"]      = df["fcf_yield"].clip(-0.50, 0.50)
        df["roic"]           = df["roic"].clip(-1.0, 3.0)

        # 6. Price momentum
        print("\n[4/6]  Multi-timeframe momentum...")
        df = add_price_momentum(df, tickers)

        # 7. Liquidity filter
        before = len(df)
        df = df[
            (df["marketCap"].fillna(0)     >= CFG["min_market_cap"]) &
            (df["averageVolume"].fillna(0) >= CFG["min_avg_volume"])
        ].copy()
        print(f"\n  Liquidity filter: {before} → {len(df)} "
              f"(removed {before - len(df)})")

        # 8. Coverage filter
        df["coverage"] = compute_coverage(df)
        before2 = len(df)
        df = df[df["coverage"] >= CFG["min_coverage"]].copy()
        print(f"  Coverage filter:  {before2} → {len(df)} "
              f"(removed {before2 - len(df)})")

        # 9. Dynamic thresholds
        print("\n[5/6]  Dynamic sector thresholds...")
        _SECTOR_THRESHOLDS = build_sector_thresholds(df)

        # 10. Pillar scores
        print("\n[6/6]  Pillar scores...")
        df = build_pillar_scores(df)

        # 11. Composite + valuation (raw composite; coverage weighting applied below)
        df["composite_score"] = df.apply(compute_composite,       axis=1)
        df["valuation_score"] = df.apply(compute_valuation_score, axis=1)

        save_cache(df)

    # 12. Coverage-weighted composite + rank + sector context (always, so cache and fresh runs match)
    # FIX B5: Old formula `composite * (0.5 + 0.5 * coverage)` double-penalized stocks
    # because compute_composite already handles missing pillars by renormalizing weights.
    # The multiplicative coverage penalty on top punished disproportionately.
    #
    # New approach: only apply penalty below a threshold (70% coverage).
    # Above 70% = no penalty. Below 70% = gentle linear discount capped at 15%.
    cov = df["coverage"].fillna(0)
    COVERAGE_THRESHOLD = 0.70
    MAX_PENALTY = 0.15  # max 15% discount for worst coverage
    coverage_factor = np.where(
        cov >= COVERAGE_THRESHOLD,
        1.0,  # no penalty
        1.0 - MAX_PENALTY * (1.0 - cov / COVERAGE_THRESHOLD)  # linear 0–15% discount
    )
    df["composite_raw"] = df["composite_score"].copy()
    df["composite_score"] = df["composite_score"] * coverage_factor
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df = add_sector_context(df)

    # 13. Output
    _print_summary(df)
    print("\n  Generating charts...")
    plot_all(df)
    print("\n  Exporting Excel...")
    style_and_export(df, CFG["output_file"])
    print("\n  Merging breakout scanner signals...")
    df = merge_breakout_signals(df)
    print("\n  Exporting JSON for web dashboard...")
    export_json(df)

    print("\n✅  DONE!")
    print(f"    Excel  → {CFG['output_file']}")
    print("    Charts → artifacts/*.png")
    return df
