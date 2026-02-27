# ranker/export_json.py — JSON export for dashboard
import json, os
from datetime import datetime
import numpy as np
import pandas as pd
from ranker.config import _TR_AVAILABLE

def export_json(df: pd.DataFrame):
    """Export ranking data as sp500_data.json for the web dashboard."""
    def safe(v):
        if v is None: return None
        try:
            f = float(v)
            return None if (f != f) else round(f, 4)   # NaN → None
        except Exception:
            return str(v) if v else None

    def pct(v):
        """Fields stored as 0-1 fraction → multiply ×100 for display."""
        r = safe(v)
        return round(r * 100, 2) if r is not None else None

    records = []
    for _, row in df.iterrows():
        records.append({
            "rank":           int(row.get("rank", 0)),
            "ticker":         str(row.get("ticker", "")),
            "company":        str(row.get("name", "")),
            "sector":         str(row.get("sector", "")),
            "industry":       str(row.get("industry", "")),
            # Core scores
            "composite":      safe(row.get("composite_score")),
            "composite_raw":  safe(row.get("composite_raw")),
            "valuation":      safe(row.get("valuation_score")),
            # Pillars
            "p_valuation":    safe(row.get("pillar_valuation")),
            "p_profitability":safe(row.get("pillar_profitability")),
            "p_growth":       safe(row.get("pillar_growth")),
            "p_earn_rev":     safe(row.get("pillar_earnings_revisions")),  # NEW pillar
            "p_eq":           safe(row.get("pillar_earnings_quality")),
            "p_fcf":          safe(row.get("pillar_fcf")),
            "p_health":       safe(row.get("pillar_health")),
            "p_momentum":     safe(row.get("pillar_momentum")),
            "p_analyst":      safe(row.get("pillar_analyst")),
            "p_piotroski":    safe(row.get("pillar_piotroski")),
            # TipRanks
            "tr_smartscore":  safe(row.get("tr_smart_score")),
            "tr_consensus":   str(row.get("tr_analyst_consensus", "") or ""),
            "tr_pt_upside":   pct(row.get("tr_pt_upside")),
            "tr_news_bull":   pct(row.get("tr_news_bullish")),
            "tr_blogger_bull":pct(row.get("tr_blogger_bullish")),
            "tr_insider":     str(row.get("tr_insider_trend", "") or ""),
            "tr_hedge":       str(row.get("tr_hedge_trend", "") or ""),
            # Valuation multiples
            "pe":             safe(row.get("trailingPE")),
            "fwd_pe":         safe(row.get("forwardPE")),
            "peg":            safe(row.get("pegRatio")),
            "ps":             safe(row.get("priceToSalesTrailing12Months")),
            "pb":             safe(row.get("priceToBook")),
            "ev_ebitda":      safe(row.get("enterpriseToEbitda")),
            # Profitability  (0-1 fraction → ×100)
            "roe":            pct(row.get("returnOnEquity")),
            "roa":            pct(row.get("returnOnAssets")),
            "roic":           pct(row.get("roic")),
            "net_margin":     pct(row.get("profitMargins")),
            "gross_margin":   pct(row.get("grossMargins")),
            "op_margin":      pct(row.get("operatingMargins")),
            # Growth  (0-1 fraction → ×100)
            "rev_growth":     pct(row.get("revenueGrowth")),
            "eps_growth":     pct(row.get("earningsGrowth")),
            # Earnings Revisions
            "earn_rev_score": safe(row.get("earnings_revision_score")),
            "rev_ratio_30d":  safe(row.get("rev_ratio_30d")),
            "eps_rev_90d":    pct(row.get("eps_revision_pct_90d")),
            # FCF  (0-1 fraction → ×100)
            "fcf_yield":      pct(row.get("fcf_yield")),
            "fcf_margin":     pct(row.get("fcf_margin")),
            # Other
            "current_ratio":  safe(row.get("currentRatio")),
            "debt_equity":    safe(row.get("debtToEquity")),
            "div_yield":      pct(row.get("dividendYield")),
            "beta":           safe(row.get("beta")),
            # Momentum  (0-1 fraction → ×100)
            "perf_12m":       pct(row.get("perf_12m")),
            "perf_6m":        pct(row.get("perf_6m")),
            "perf_3m":        pct(row.get("perf_3m")),
            "perf_1m":        pct(row.get("perf_1m")),
            "momentum":       pct(row.get("momentum_composite")),
            "rs_12m":         pct(row.get("rs_12m")),
            "rs_6m":          pct(row.get("rs_6m")),
            "rs_3m":          pct(row.get("rs_3m")),
            "rs_1m":          pct(row.get("rs_1m")),
            "rs_composite":   pct(row.get("rs_composite")),
            "p_rs":           safe(row.get("pillar_relative_strength")),
            # Risk
            "altman_z":       safe(row.get("altman_z")),
            "piotroski_f":    safe(row.get("piotroski_score")),
            "eq_score":       safe(row.get("earnings_quality_score")),
            # Market data
            "price":          safe(row.get("currentPrice")),
            "market_cap":     safe(row.get("marketCap")),
            "avg_volume":     safe(row.get("averageVolume")),
            "pt_upside":      pct(row.get("pt_upside")),
            "analysts":       safe(row.get("numberOfAnalystOpinions")),
            "analyst_mean":   safe(row.get("recommendationMean")),
            "payout_ratio":   pct(row.get("payoutRatio")),
            "fcf_to_ni":      safe(row.get("fcf_to_ni")),
            "tr_asset_gr":    pct(row.get("tr_asset_growth")),
            "tr_inv_chg_30d": pct(row.get("tr_investor_chg_30d")),
            "tr_inv_chg_7d":  pct(row.get("tr_investor_chg_7d")),
            "tr_mom_12m":     pct(row.get("tr_momentum_12m")),
            # FIX v5.3: New signals
            "short_ratio":    safe(row.get("shortRatio")),
            "insider_pct_mcap": pct(row.get("insider_pct_mcap")),
            "ma_regime":      safe(row.get("ma_regime_score")),
            "pct_above_ma200": pct(row.get("pct_above_ma200")),
            "coverage":       pct(row.get("coverage")),
            "vs_sector":      safe(row.get("vs_sector")),
            # Breakout scanner
            "breakout_score":  safe(row.get("breakout_score")),
            "breakout_rank":   safe(row.get("breakout_rank")),
            "breakout_phase":  safe(row.get("breakout_phase")),
            "breakout_rr":     safe(row.get("breakout_rr")),
            "breakout_rs":     safe(row.get("breakout_rs")),
            "breakout_stop":   safe(row.get("breakout_stop")),
            "has_vcp":         bool(row.get("has_vcp")) if row.get("has_vcp") else False,
            "vcp_quality":     safe(row.get("vcp_quality")),
            "breakout_entry":  str(row.get("breakout_entry_quality","") or ""),
            "breakout_reasons":str(row.get("breakout_reasons","") or ""),
        })

    import json as _json
    payload = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        "count":     len(records),
        "tr_available": _TR_AVAILABLE,
        "weights_mode": "standard" if _TR_AVAILABLE else "no_tipranks_fallback",
        "data":      records,
    }
    json_path = "artifacts/sp500_data.json"
    with open(json_path, "w") as f:
        _json.dump(payload, f, separators=(",", ":"))
    with open("sp500_data.json", "w") as f:
        _json.dump(payload, f, separators=(",", ":"))
    n_rs = sum(1 for r in records if r.get("rs_12m") is not None)
    print(f"✅  JSON → {json_path}  ({len(records)} stocks, {n_rs} with RS vs SPY)")
