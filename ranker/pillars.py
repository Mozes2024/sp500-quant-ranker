# ranker/pillars.py — Pillar score construction + sector percentile
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from ranker.config import CFG, GLOBAL_THRESHOLDS, _SECTOR_THRESHOLDS

def sector_percentile(df: pd.DataFrame, col: str,
                      higher_is_better: bool = True) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for _, grp in df.groupby("sector"):
        valid = grp[col].dropna()
        if valid.empty:
            continue
        pct = grp[col].apply(
            lambda x: percentileofscore(valid, x, kind="rank") / 100
            if not pd.isna(x) else np.nan
        )
        if not higher_is_better:
            pct = 1 - pct
        result.loc[grp.index] = pct * 90 + 10
    return result


# ════════════════════════════════════════════════════════════
#  PILLAR SCORES
# ════════════════════════════════════════════════════════════

def build_pillar_scores(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Valuation
    df["s_pe"]        = sector_percentile(df, "trailingPE",                  False)
    df["s_peg"]       = sector_percentile(df, "pegRatio",                    False)
    df["s_ev_ebitda"] = sector_percentile(df, "enterpriseToEbitda",           False)
    df["s_ps"]        = sector_percentile(df, "priceToSalesTrailing12Months", False)
    df["s_pb"]        = sector_percentile(df, "priceToBook",                  False)
    df["pillar_valuation"] = df[["s_pe","s_peg","s_ev_ebitda","s_ps","s_pb"]].mean(axis=1, skipna=True)

    # 2. Profitability
    df["s_roe"]    = sector_percentile(df, "returnOnEquity", True)
    df["s_roa"]    = sector_percentile(df, "returnOnAssets", True)
    df["s_roic"]   = sector_percentile(df, "roic",           True)
    df["s_pm"]     = sector_percentile(df, "profitMargins",  True)
    df["s_tr_roe"] = sector_percentile(df, "tr_roe",         True)
    df["pillar_profitability"] = df[["s_roe","s_roa","s_roic","s_pm","s_tr_roe"]].mean(axis=1, skipna=True)

    # 3. Growth — TRIMMED: earnings revision signals moved to dedicated pillar (3b)
    #            Now: pure realized growth + forward-looking asset growth
    df["s_rev_g"]        = sector_percentile(df, "revenueGrowth",            True)
    df["s_earn_g"]       = sector_percentile(df, "earningsGrowth",           True)
    df["s_tr_asset_g"]   = sector_percentile(df, "tr_asset_growth",          False)  # high asset growth → lower future returns (Titman et al. 2004)
    df["pillar_growth"]  = df[["s_rev_g","s_earn_g","s_tr_asset_g"]].mean(axis=1, skipna=True)

    # 3b. Earnings Revisions — NEW INDEPENDENT PILLAR
    #     Research: Zacks 1979, Chan/Jegadeesh/Lakonishok 1996
    #     Earnings revisions are among the strongest short-term alpha factors.
    #     Previously buried inside Growth pillar with only 20% effective weight.
    #     Now standalone at 10% of composite — gives revisions the weight they deserve.
    #     Signals:
    #       - earn_rev_score: breadth (up vs down 30d) + magnitude (EPS change 90d) + acceleration (7d)
    #       - eps_revision_pct_30d: short-term EPS estimate velocity
    #       - rev_ratio_30d: raw revision breadth ratio (-1 to +1)
    df["s_earn_rev"]     = sector_percentile(df, "earnings_revision_score",  True)
    df["s_eps_rev_30d"]  = sector_percentile(df, "eps_revision_pct_30d",     True)
    df["s_rev_ratio"]    = sector_percentile(df, "rev_ratio_30d",            True)
    df["pillar_earnings_revisions"] = df[["s_earn_rev","s_eps_rev_30d","s_rev_ratio"]].mean(axis=1, skipna=True)

    # 4. Earnings Quality
    df["s_eq"] = sector_percentile(df, "earnings_quality_score", True)
    df["pillar_earnings_quality"] = df["s_eq"]

    # 5. FCF Quality
    df["s_fcf_yield"] = sector_percentile(df, "fcf_yield",  True)
    df["s_fcf_ni"]    = sector_percentile(df, "fcf_to_ni",  True)
    df["s_fcf_m"]     = sector_percentile(df, "fcf_margin", True)   # FIX: re-added here (was removed from growth to avoid double-count, but still belongs in FCF pillar)
    df["pillar_fcf"]  = df[["s_fcf_yield","s_fcf_ni","s_fcf_m"]].mean(axis=1, skipna=True)

    # 6. Financial Health — added beta (low volatility = financial stability signal)
    df["s_cr"]     = sector_percentile(df, "currentRatio", True)
    df["s_de"]     = sector_percentile(df, "debtToEquity", False)
    df["s_altman"] = sector_percentile(df, "altman_z",     True)
    df["s_beta"]   = sector_percentile(df, "beta",         False)  # FIX: moved from momentum — low vol is a risk/health signal, not momentum
    df["pillar_health"] = df[["s_cr","s_de","s_altman","s_beta"]].mean(axis=1, skipna=True)

    # 7. Momentum — pure price-momentum + MA regime + short squeeze signal
    df["s_mom"]        = sector_percentile(df, "momentum_composite", True)
    df["s_tr_mom12"]   = sector_percentile(df, "tr_momentum_12m",    True)
    df["s_tr_sma"]     = sector_percentile(df, "tr_sma_num",         True)
    df["s_short"]      = sector_percentile(df, "shortRatio",         False)  # low short ratio = bullish
    df["s_ma_regime"]  = sector_percentile(df, "ma_regime_score",    True)   # NEW: sweet-spot MA200 signal
    df["pillar_momentum"] = df[["s_mom","s_tr_mom12","s_tr_sma","s_short","s_ma_regime"]].mean(axis=1, skipna=True)

    # 8. Analyst — REVIVED with clean composition (no SmartScore to avoid double-count)
    # Signals:
    #   - Price Target Upside (avg Yahoo + TR): unique forward-looking signal (40%)
    #   - Insider buying as % MCap: skin-in-the-game, not in any other pillar (35%)
    #   - Yahoo Recommendation Mean: broad consensus fallback (25%)
    #   - SmartScore intentionally EXCLUDED — already baked into tr_roe, tr_momentum, etc.
    df["s_rec"]          = sector_percentile(df, "recommendationMean",   False)
    df["s_pt_upside"]    = sector_percentile(df, "pt_upside",            True)
    df["s_tr_pt"]        = sector_percentile(df, "tr_pt_upside",         True)
    df["s_insider_mcap"] = sector_percentile(df, "insider_pct_mcap",     True)
    # PT: average of Yahoo and TipRanks to avoid single-source bias
    s_pt_avg = df[["s_pt_upside","s_tr_pt"]].mean(axis=1, skipna=True)
    df["pillar_analyst"] = (
        0.40 * s_pt_avg.fillna(df["s_pt_upside"].fillna(50)) +
        0.35 * df["s_insider_mcap"].fillna(50) +
        0.25 * df["s_rec"].fillna(50)
    )
    # Rescale to 10–100 band (same as sector_percentile output)
    _min, _max = df["pillar_analyst"].min(), df["pillar_analyst"].max()
    if _max > _min:
        df["pillar_analyst"] = ((df["pillar_analyst"] - _min) / (_max - _min)) * 90 + 10

    # SmartScore: still computed for display in dashboard, but NOT in any weighted pillar
    df["s_tr_smart"]     = sector_percentile(df, "tr_smart_score",       True)

    # 9. Piotroski — DISPLAY ONLY (no weight in composite, removed from PILLAR_MAP)
    # Still computed for reference in detail panels and Excel export
    df["s_piotroski"]      = sector_percentile(df, "piotroski_score", True)
    df["pillar_piotroski"] = df["s_piotroski"]

    # 10. Relative strength vs market (excess return vs SPY)
    df["s_rs_12m"] = sector_percentile(df, "rs_12m", True)
    df["s_rs_6m"]  = sector_percentile(df, "rs_6m",  True)
    df["s_rs_3m"]  = sector_percentile(df, "rs_3m",  True)
    df["pillar_relative_strength"] = df[["s_rs_12m", "s_rs_6m", "s_rs_3m"]].mean(axis=1, skipna=True)

    return df
