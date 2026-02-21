# ============================================================
#  S&P 500 ADVANCED RANKING SYSTEM v5.2 – GitHub Edition
#  Full integration with MOZES Stock Scanner (breakout_signals.json)
#  Cache fix + fresh run by default
# ============================================================

import subprocess, sys, os, pickle, time, shutil
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "pandas", "numpy", "openpyxl==3.1.2", "requests", "beautifulsoup4", "matplotlib", "seaborn", "tqdm", "scipy", "-q"])

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

# ========================== CONFIG ==========================
CFG = {
    "weights": {
        "valuation": 0.19, "profitability": 0.16, "growth": 0.13,
        "earnings_quality": 0.08, "fcf_quality": 0.13, "financial_health": 0.09,
        "momentum": 0.09, "analyst": 0.11, "piotroski": 0.02,
    },
    "min_coverage": 0.45,
    "min_market_cap": 5_000_000_000,
    "min_avg_volume": 500_000,
    "cache_hours": 24,
    "sleep_tr": 0.35,
    "batch_size_tr": 10,
    "max_workers_yf": 20,
    "output_file": "artifacts/sp500_ranking_v5.2.xlsx",
}
assert abs(sum(CFG["weights"].values()) - 1.0) < 1e-6

CACHE_FILE = "sp500_cache_v5.pkl"
_SECTOR_THRESHOLDS = {}

# ========================== TIPRANKS ==========================
TR_URL = "https://mobile.tipranks.com/api/stocks/stockAnalysisOverview"
TR_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.tipranks.com/",
    "Origin": "https://www.tipranks.com",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}
# ... (כל הפונקציות TIPRANKS, _parse_tipranks, fetch_tipranks – בדיוק כמו בגרסה שלך)

# ========================== S&P TICKERS ==========================
def get_sp500_tickers():
    # (כל 3 האסטרטגיות – Wikipedia + read_html + CSV – בדיוק כמו שהיה)
    # ...

# ========================== YAHOO FINANCE ==========================
FUNDAMENTAL_FIELDS = [ ... ]  # (כל הרשימה כמו שהיה)

def fetch_yf_parallel(tickers):
    # (הקוד המלא של parallel fetch)

def add_price_momentum(df, tickers):
    # (הקוד המלא)

# ========================== COMPUTED METRICS ==========================
def compute_piotroski(row): ...
def compute_altman(row): ...
def compute_roic(row): ...
def compute_fcf_metrics(row): ...
def compute_earnings_quality(row): ...
def compute_pt_upside(row): ...
def compute_tr_pt_upside(row): ...

# ========================== SECTOR THRESHOLDS & NORMALISATION ==========================
GLOBAL_THRESHOLDS = { ... }
def build_sector_thresholds(df): ...
def sector_percentile(df, col, higher_is_better=True): ...

# ========================== PILLAR SCORES ==========================
def build_pillar_scores(df): ...
def compute_composite(row): ...
def compute_valuation_score(row): ...

# ========================== COVERAGE ==========================
def compute_coverage(df): ...
def add_sector_context(df): ...

# ========================== CACHE ==========================
def load_cache():
    global _SECTOR_THRESHOLDS
    if not os.path.exists(CACHE_FILE): return None
    try:
        with open(CACHE_FILE, "rb") as f:
            payload = pickle.load(f)
        if len(payload) == 3:
            data, saved_thresholds, ts = payload
            _SECTOR_THRESHOLDS = saved_thresholds
        else:
            data, ts = payload
            _SECTOR_THRESHOLDS = build_sector_thresholds(data)
        age = datetime.now() - ts
        if age < timedelta(hours=CFG["cache_hours"]):
            print(f"✅ Cache loaded ({int(age.total_seconds()//60)} min old)")
            return data
    except Exception as e:
        print(f"⚠ Cache read error: {e}")
    return None

def save_cache(df):
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump((df, _SECTOR_THRESHOLDS, datetime.now()), f)
        print(f"💾 Cache saved")
    except Exception as e:
        print(f"⚠ Cache save error: {e}")

# ========================== EXPORT & PLOTS ==========================
def style_and_export(df, filepath): ...
def plot_all(df): ...
def _plot_radar(df_top): ...

# ========================== MERGE BREAKOUT (מתוקן) ==========================
def merge_breakout_signals(df: pd.DataFrame) -> pd.DataFrame:
    import json as _json, os as _os
    bp = "breakout_signals.json"
    if not _os.path.exists(bp):
        print("  ⚠ breakout_signals.json not found — skipping")
        for col in ["breakout_score","breakout_rank","breakout_phase","breakout_rr","breakout_rs","breakout_stop","has_vcp","vcp_quality","breakout_entry_quality","breakout_reasons"]:
            df[col] = np.nan
        df["has_vcp"] = False
        df["breakout_entry_quality"] = ""
        df["breakout_reasons"] = ""
        return df

    with open(bp, encoding="utf-8") as f:
        data = _json.load(f)

    signals = {s["ticker"]: s for s in data.get("top_signals", [])}
    print(f"  ✅ Loaded {len(signals)} breakout signals")

    def _get(t, field, default=np.nan):
        return signals.get(t, {}).get(field, default)

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

    print(f"  🔀 Overlap: {df['breakout_score'].notna().sum()} stocks")
    return df

# ========================== JSON EXPORT ==========================
def export_json(df: pd.DataFrame):
    def safe(v):
        if v is None: return None
        try:
            f = float(v)
            return None if (f != f) else round(f, 4)
        except:
            return str(v) if v else None

    records = []
    for _, row in df.iterrows():
        records.append({
            "rank": int(row.get("rank", 0)),
            "ticker": str(row.get("ticker", "")),
            "company": str(row.get("name", "")),
            "sector": str(row.get("sector", "")),
            "industry": str(row.get("industry", "")),
            "composite": safe(row.get("composite_score")),
            "valuation": safe(row.get("valuation_score")),
            "p_valuation": safe(row.get("pillar_valuation")),
            "p_profitability": safe(row.get("pillar_profitability")),
            "p_growth": safe(row.get("pillar_growth")),
            "p_eq": safe(row.get("pillar_earnings_quality")),
            "p_fcf": safe(row.get("pillar_fcf")),
            "p_health": safe(row.get("pillar_health")),
            "p_momentum": safe(row.get("pillar_momentum")),
            "p_analyst": safe(row.get("pillar_analyst")),
            "p_piotroski": safe(row.get("pillar_piotroski")),
            "tr_smartscore": safe(row.get("tr_smart_score")),
            "breakout_score": safe(row.get("breakout_score")),
            "breakout_rank": safe(row.get("breakout_rank")),
            "breakout_phase": safe(row.get("breakout_phase")),
            "breakout_rr": safe(row.get("breakout_rr")),
            "breakout_rs": safe(row.get("breakout_rs")),
            "breakout_stop": safe(row.get("breakout_stop")),
            "has_vcp": bool(row.get("has_vcp")),
            "vcp_quality": safe(row.get("vcp_quality")),
            "breakout_entry": str(row.get("breakout_entry_quality","")),
            "breakout_reasons": str(row.get("breakout_reasons","")),
            # כל השדות האחרים שלך – הוסף כאן אם חסרים
        })

    payload = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M UTC"), "count": len(records), "data": records}

    with open("artifacts/sp500_data.json", "w") as f:
        import json
        json.dump(payload, f, separators=(",", ":"))
    with open("sp500_data.json", "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    print(f"✅ JSON exported ({len(records)} stocks with breakout)")

# ========================== RUN PIPELINE ==========================
def run_pipeline(use_cache: bool = False) -> pd.DataFrame:   # False = fresh run
    global _SECTOR_THRESHOLDS
    df = None

    print("=" * 65)
    print(f"  S&P 500 ADVANCED RANKING v5.2 – GitHub Edition")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    cached = load_cache() if use_cache else None
    if cached is not None:
        df = cached
        print("  ✅ Using cached data")
    else:
        # ====================== כאן כל הקוד המקורי שלך ======================
        # get_sp500_tickers, fetch, compute_piotroski, build_pillar_scores, add_sector_context, save_cache וכו'
        # (העתק מהגרסה הישנה שלך את כל הבלוק הזה – הוא חייב להיות כאן)
        universe = get_sp500_tickers()
        tickers = universe["ticker"].tolist()
        yf_data = fetch_yf_parallel(tickers)
        fund_df = pd.DataFrame.from_dict(yf_data, orient="index").reset_index()
        df = universe.merge(fund_df, on="ticker", how="left")
        tr_df = fetch_tipranks(tickers)
        if not tr_df.empty:
            df = df.merge(tr_df, on="ticker", how="left")
        df["piotroski_score"] = df.apply(compute_piotroski, axis=1)
        df["altman_z"] = df.apply(compute_altman, axis=1)
        df["roic"] = df.apply(compute_roic, axis=1)
        fcf_cols = df.apply(compute_fcf_metrics, axis=1, result_type="expand")
        df = pd.concat([df, fcf_cols], axis=1)
        df["pt_upside"] = df.apply(compute_pt_upside, axis=1)
        df["tr_pt_upside"] = df.apply(compute_tr_pt_upside, axis=1)
        df["earnings_quality_score"] = df.apply(compute_earnings_quality, axis=1)
        df = add_price_momentum(df, tickers)
        df = df[df["marketCap"].fillna(0) >= CFG["min_market_cap"]]
        df = df[df["averageVolume"].fillna(0) >= CFG["min_avg_volume"]]
        df["coverage"] = compute_coverage(df)
        df = df[df["coverage"] >= CFG["min_coverage"]]
        _SECTOR_THRESHOLDS = build_sector_thresholds(df)
        df = build_pillar_scores(df)
        df["composite_score"] = df.apply(compute_composite, axis=1)
        df["valuation_score"] = df.apply(compute_valuation_score, axis=1)
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        df = add_sector_context(df)
        save_cache(df)

    if df is None:
        raise RuntimeError("Critical error: df was never created!")

    print("\n  Merging breakout scanner signals...")
    df = merge_breakout_signals(df)

    print("\n  Exporting JSON for web dashboard...")
    export_json(df)

    print("\n✅ DONE! Everything worked.")
    return df

# ========================== ENTRY POINT ==========================
if __name__ == "__main__":
    run_pipeline(use_cache=False)   # False = fresh run – חובה!
