# ============================================================
#  S&P 500 ADVANCED RANKING SYSTEM  v5.2 – GitHub Edition
#  Headless · Actions Cache · Parallel Fetch · Full Pipeline
#  Integrates with MOZES Stock Scanner (breakout_signals.json)
# ============================================================

import subprocess, sys, os, pickle, time, shutil
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "yfinance", "pandas", "numpy", "openpyxl==3.1.2",
    "requests", "beautifulsoup4", "matplotlib", "seaborn",
    "tqdm", "scipy", "-q",
])

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
from bs4 import BeautifulSoup
from tqdm import tqdm
from scipy.stats import percentileofscore
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.2f}".format)

os.makedirs("artifacts", exist_ok=True)


# ════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════
CFG = {
    "weights": {
        "valuation":        0.19,
        "profitability":    0.16,
        "growth":           0.13,
        "earnings_quality": 0.08,
        "fcf_quality":      0.13,
        "financial_health": 0.09,
        "momentum":         0.09,
        "analyst":          0.11,
        "piotroski":        0.02,
    },
    "min_coverage":    0.45,
    "min_market_cap":  5_000_000_000,
    "min_avg_volume":  500_000,
    "cache_hours":     24,
    "sleep_tr":        0.35,
    "batch_size_tr":   10,
    "max_workers_yf":  20,
    "output_file":     "artifacts/sp500_ranking_v5.2.xlsx",
}
assert abs(sum(CFG["weights"].values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

CACHE_FILE = "sp500_cache_v5.pkl"


# (כל שאר הפונקציות – TIPRANKS, get_sp500_tickers, fetch_yf_parallel וכו' – נשארו בדיוק כמו שהיו)

# ... [כל הקוד עד לפני merge_breakout_signals] ...


# ════════════════════════════════════════════════════════════
#  MERGE BREAKOUT SIGNALS (מתוקן ומלא)
# ════════════════════════════════════════════════════════════
def merge_breakout_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge breakout scan signals from MOZES Automated Stock Scanner.
    Reads breakout_signals.json (pushed by the scanner workflow).
    Adds all breakout columns to the main ranking.
    """
    import json as _json, os as _os
    bp = "breakout_signals.json"

    if not _os.path.exists(bp):
        print("  ⚠  breakout_signals.json not found — skipping breakout merge")
        for col in ["breakout_score","breakout_rank","breakout_phase",
                    "breakout_rr","breakout_rs","breakout_stop",
                    "has_vcp","vcp_quality","breakout_entry_quality","breakout_reasons"]:
            df[col] = np.nan
        df["has_vcp"] = False
        df["breakout_entry_quality"] = ""
        df["breakout_reasons"] = ""
        return df

    with open(bp, encoding="utf-8") as f:
        data = _json.load(f)

    signals = {s["ticker"]: s for s in data.get("top_signals", [])}
    print(f"  ✅ Loaded {len(signals)} breakout signals (scan: {data.get('scan_date','')})")

    def _get(ticker, field, default=np.nan):
        return signals.get(ticker, {}).get(field, default)

    # חשוב: משתמש ב-"ticker" (t קטנה) כמו בכל השאר של הסקריפט
    df["breakout_score"]         = df["ticker"].apply(lambda t: _get(t, "breakout_score"))
    df["breakout_rank"]          = df["ticker"].apply(lambda t: _get(t, "rank"))
    df["breakout_phase"]         = df["ticker"].apply(lambda t: _get(t, "phase"))
    df["breakout_rr"]            = df["ticker"].apply(lambda t: _get(t, "risk_reward"))
    df["breakout_rs"]            = df["ticker"].apply(lambda t: _get(t, "rs"))
    df["breakout_stop"]          = df["ticker"].apply(lambda t: _get(t, "stop_loss"))
    df["has_vcp"]                = df["ticker"].apply(lambda t: signals.get(t, {}).get("has_vcp", False))
    df["vcp_quality"]            = df["ticker"].apply(lambda t: _get(t, "vcp_quality"))
    df["breakout_entry_quality"] = df["ticker"].apply(lambda t: signals.get(t, {}).get("entry_quality", ""))
    df["breakout_reasons"]       = df["ticker"].apply(lambda t: " | ".join(signals.get(t, {}).get("reasons", [])))

    n_overlap = df["breakout_score"].notna().sum()
    print(f"  🔀 Overlap with S&P 500: {n_overlap} stocks have breakout signals")
    return df


# ════════════════════════════════════════════════════════════
#  JSON EXPORT (לדאשבורד)
# ════════════════════════════════════════════════════════════
def export_json(df: pd.DataFrame):
    """Export ranking + breakout data as sp500_data.json"""
    def safe(v):
        if v is None: return None
        try:
            f = float(v)
            return None if (f != f) else round(f, 4)
        except Exception:
            return str(v) if v else None

    def pct(v):
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
            "composite":      safe(row.get("composite_score")),
            "valuation":      safe(row.get("valuation_score")),
            "p_valuation":    safe(row.get("pillar_valuation")),
            "p_profitability":safe(row.get("pillar_profitability")),
            "p_growth":       safe(row.get("pillar_growth")),
            "p_eq":           safe(row.get("pillar_earnings_quality")),
            "p_fcf":          safe(row.get("pillar_fcf")),
            "p_health":       safe(row.get("pillar_health")),
            "p_momentum":     safe(row.get("pillar_momentum")),
            "p_analyst":      safe(row.get("pillar_analyst")),
            "p_piotroski":    safe(row.get("pillar_piotroski")),
            "tr_smartscore":  safe(row.get("tr_smart_score")),
            "tr_consensus":   str(row.get("tr_analyst_consensus", "") or ""),
            "tr_pt_upside":   pct(row.get("tr_pt_upside")),
            "tr_news_bull":   pct(row.get("tr_news_bullish")),
            "tr_blogger_bull":pct(row.get("tr_blogger_bullish")),
            "tr_insider":     str(row.get("tr_insider_trend", "") or ""),
            "tr_hedge":       str(row.get("tr_hedge_trend", "") or ""),
            "pe":             safe(row.get("trailingPE")),
            "fwd_pe":         safe(row.get("forwardPE")),
            "peg":            safe(row.get("pegRatio")),
            "ps":             safe(row.get("priceToSalesTrailing12Months")),
            "pb":             safe(row.get("priceToBook")),
            "ev_ebitda":      safe(row.get("enterpriseToEbitda")),
            "roe":            pct(row.get("returnOnEquity")),
            "roa":            pct(row.get("returnOnAssets")),
            "roic":           pct(row.get("roic")),
            "net_margin":     pct(row.get("profitMargins")),
            "gross_margin":   pct(row.get("grossMargins")),
            "op_margin":      pct(row.get("operatingMargins")),
            "rev_growth":     pct(row.get("revenueGrowth")),
            "eps_growth":     pct(row.get("earningsGrowth")),
            "fcf_yield":      pct(row.get("fcf_yield")),
            "fcf_margin":     pct(row.get("fcf_margin")),
            "current_ratio":  safe(row.get("currentRatio")),
            "debt_equity":    safe(row.get("debtToEquity")),
            "div_yield":      pct(row.get("dividendYield")),
            "beta":           safe(row.get("beta")),
            "perf_12m":       pct(row.get("perf_12m")),
            "perf_6m":        pct(row.get("perf_6m")),
            "perf_3m":        pct(row.get("perf_3m")),
            "perf_1m":        pct(row.get("perf_1m")),
            "momentum":       pct(row.get("momentum_composite")),
            "altman_z":       safe(row.get("altman_z")),
            "piotroski_f":    safe(row.get("piotroski_score")),
            "eq_score":       safe(row.get("earnings_quality_score")),
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
            "coverage":       pct(row.get("coverage")),
            "vs_sector":      safe(row.get("vs_sector")),
            # Breakout fields
            "breakout_score":   safe(row.get("breakout_score")),
            "breakout_rank":    safe(row.get("breakout_rank")),
            "breakout_phase":   safe(row.get("breakout_phase")),
            "breakout_rr":      safe(row.get("breakout_rr")),
            "breakout_rs":      safe(row.get("breakout_rs")),
            "breakout_stop":    safe(row.get("breakout_stop")),
            "has_vcp":          bool(row.get("has_vcp")),
            "vcp_quality":      safe(row.get("vcp_quality")),
            "breakout_entry":   str(row.get("breakout_entry_quality","") or ""),
            "breakout_reasons": str(row.get("breakout_reasons","") or ""),
        })

    import json as _json
    payload = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        "count":     len(records),
        "data":      records,
    }

    # שומר גם בתיקיית artifacts וגם בשורש (ל-GitHub Pages)
    json_path = "artifacts/sp500_data.json"
    with open(json_path, "w") as f:
        _json.dump(payload, f, separators=(",", ":"))
    with open("sp500_data.json", "w") as f:
        _json.dump(payload, f, separators=(",", ":"))

    print(f"✅  JSON → {json_path}  ({len(records)} stocks with breakout data)")


# ════════════════════════════════════════════════════════════
#  RUN PIPELINE
# ════════════════════════════════════════════════════════════
def run_pipeline(use_cache: bool = True) -> pd.DataFrame:
    # ... (כל הקוד של run_pipeline נשאר בדיוק כמו שהיה עד שורה 1050 בערך) ...
    # בסוף הפונקציה, לפני return df, יש את השורות הבאות:

    print("\n  Merging breakout scanner signals...")
    df = merge_breakout_signals(df)          # ← כאן זה משולב

    print("\n  Exporting JSON for web dashboard...")
    export_json(df)

    print("\n✅  DONE!")
    print(f"    Excel  → {CFG['output_file']}")
    print("    Charts → artifacts/*.png")
    print("    JSON   → sp500_data.json (with breakout signals)")
    return df


# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_pipeline(use_cache=True)
