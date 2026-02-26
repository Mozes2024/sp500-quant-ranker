# ============================================================
#  S&P 500 ADVANCED RANKING SYSTEM  v5.3 – Logic Fix & Final Weights
#  26 Feb 2026 – Double-counting fixed, Analyst/Piotroski → 0%
#  Momentum + Short Ratio + Insider % Market Cap
#  Growth + eps_revision_pct_30d (replaced fcf_to_ni)
#  Headless · Actions Cache · Parallel Fetch · Full Pipeline
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
#  CONFIGURATION – v5.3 FINAL WEIGHTS (26/02/2026)
# ════════════════════════════════════════════════════════════
CFG = {
    "weights": {
        "valuation":         0.15,   # v5.3
        "profitability":     0.18,   # v5.3
        "growth":            0.13,   # v5.3
        "earnings_quality":  0.09,   # v5.3
        "fcf_quality":       0.13,   # v5.3
        "financial_health":  0.10,   # v5.3
        "momentum":          0.10,   # v5.3 (+ Short + Insider %)
        "relative_strength": 0.12,   # v5.3
        "analyst":           0.00,   # v5.3 – ZEROED (SmartScore double-count risk)
        "piotroski":         0.00,   # v5.3 – ZEROED (proxy noise)
    },
    "min_coverage":    0.45,
    "coverage_composite_min": 0.5,
    "min_market_cap":  5_000_000_000,
    "min_avg_volume":  500_000,
    "cache_hours":     24,
    "sleep_tr":        0.35,
    "batch_size_tr":   10,
    "max_workers_yf":  20,
    "output_file":     "artifacts/sp500_ranking_v5.3.xlsx",   # v5.3
}
assert abs(sum(CFG["weights"].values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

CACHE_FILE = "sp500_cache_v11.pkl"  # v5.3 bumped for final logic


# ════════════════════════════════════════════════════════════
#  TIPRANKS (ללא שינוי)
# ════════════════════════════════════════════════════════════
TR_URL = "https://mobile.tipranks.com/api/stocks/stockAnalysisOverview"
TR_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         "https://www.tipranks.com/",
    "Origin":          "https://www.tipranks.com",
    "Connection":      "keep-alive",
    "Sec-Fetch-Dest":  "empty",
    "Sec-Fetch-Mode":  "cors",
    "Sec-Fetch-Site":  "same-site",
}

_CONSENSUS = {
    "StrongBuy": 5, "Buy": 4, "Moderate Buy": 3.5,
    "Hold": 3, "Moderate Sell": 2, "Sell": 1, "StrongSell": 0,
}
_TREND = {
    "Increased": 1, "Unchanged": 0, "Decreased": -1,
    "BoughtShares": 1, "SoldShares": -1,
}
_SENTIMENT = {
    "VeryBullish": 5, "Bullish": 4, "Neutral": 3,
    "Bearish": 2, "VeryBearish": 1,
    "VeryPositive": 5, "Positive": 4,
    "Negative": 2, "VeryNegative": 1,
}
_SMA = {"Positive": 1, "Neutral": 0, "Negative": -1}


def _parse_tipranks(item: dict) -> dict:
    return {
        "tr_smart_score":        item.get("smartScore"),
        "tr_price_target":       item.get("convertedPriceTarget"),
        "tr_insider_3m_usd":     item.get("insidersLast3MonthsSum"),
        "tr_hedge_fund_value":   item.get("hedgeFundTrendValue"),
        "tr_investor_chg_7d":    item.get("investorHoldingChangeLast7Days"),
        "tr_investor_chg_30d":   item.get("investorHoldingChangeLast30Days"),
        "tr_momentum_12m":       item.get("technicalsTwelveMonthsMomentum"),
        "tr_roe":                item.get("fundamentalsReturnOnEquity"),
        "tr_asset_growth":       item.get("fundamentalsAssetGrowth"),
        "tr_blogger_bullish":    item.get("bloggerBullishSentiment"),
        "tr_blogger_sector_avg": item.get("bloggerSectorAvg"),
        "tr_news_bullish":       item.get("newsSentimentsBullishPercent"),
        "tr_news_bearish":       item.get("newsSentimentsBearishPercent"),
        "tr_consensus_num":      _CONSENSUS.get(item.get("analystConsensus"),  np.nan),
        "tr_hedge_trend_num":    _TREND.get(item.get("hedgeFundTrend"),        np.nan),
        "tr_insider_trend_num":  _TREND.get(item.get("insiderTrend"),          np.nan),
        "tr_news_sent_num":      _SENTIMENT.get(item.get("newsSentiment"),     np.nan),
        "tr_blogger_cons_num":   _SENTIMENT.get(item.get("bloggerConsensus"),  np.nan),
        "tr_investor_sent_num":  _SENTIMENT.get(item.get("investorSentiment"), np.nan),
        "tr_sma_num":            _SMA.get(item.get("sma"),                     np.nan),
        "tr_analyst_consensus":  item.get("analystConsensus"),
        "tr_hedge_trend":        item.get("hedgeFundTrend"),
        "tr_insider_trend":      item.get("insiderTrend"),
        "tr_news_sentiment":     item.get("newsSentiment"),
        "tr_sma":                item.get("sma"),
    }


def fetch_tipranks(tickers: list) -> pd.DataFrame:
    results = {}
    _TR_COLS = list(_parse_tipranks({}).keys())
    chunks = [tickers[i:i + CFG["batch_size_tr"]]
              for i in range(0, len(tickers), CFG["batch_size_tr"])]

    for chunk in tqdm(chunks, desc="TipRanks"):
        try:
            resp = requests.get(TR_URL,
                                params={"tickers": ",".join(chunk)},
                                headers=TR_HEADERS, timeout=15)
            if resp.status_code == 200:
                for item in resp.json():
                    t = item.get("ticker", "")
                    if t:
                        results[t] = _parse_tipranks(item)
            else:
                print(f"  ⚠️  TipRanks HTTP {resp.status_code}")
        except Exception as e:
            print(f"  ⚠️  TipRanks error: {e}")
        time.sleep(CFG["sleep_tr"])

    if not results:
        print("  ⚠️  TipRanks returned no data — all TR columns will be NaN.")
        return pd.DataFrame(columns=["ticker"] + _TR_COLS)

    tr_df = pd.DataFrame.from_dict(results, orient="index")
    tr_df.index.name = "ticker"
    tr_df = tr_df.reset_index()
    for col in _TR_COLS:
        if col not in tr_df.columns:
            tr_df[col] = np.nan
    print(f"  ✅  TipRanks: {tr_df['tr_smart_score'].notna().sum()}/{len(tickers)} SmartScores")
    return tr_df


# (כל שאר הפונקציות – get_sp500_tickers, _get_one, fetch_yf_parallel, add_price_momentum,
# compute_piotroski, compute_altman, compute_roic, compute_fcf_metrics, compute_earnings_quality,
# compute_earnings_revision_score, build_sector_thresholds, sector_percentile – ללא שינוי)

# ════════════════════════════════════════════════════════════
#  PILLAR SCORES – v5.3 (key changes marked)
# ════════════════════════════════════════════════════════════
def build_pillar_scores(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Valuation (ללא שינוי)
    df["s_pe"]        = sector_percentile(df, "trailingPE",                  False)
    df["s_peg"]       = sector_percentile(df, "pegRatio",                    False)
    df["s_ev_ebitda"] = sector_percentile(df, "enterpriseToEbitda",           False)
    df["s_ps"]        = sector_percentile(df, "priceToSalesTrailing12Months", False)
    df["s_pb"]        = sector_percentile(df, "priceToBook",                  False)
    df["pillar_valuation"] = df[["s_pe","s_peg","s_ev_ebitda","s_ps","s_pb"]].mean(axis=1, skipna=True)

    # 2. Profitability (ללא שינוי)
    df["s_roe"]    = sector_percentile(df, "returnOnEquity", True)
    df["s_roa"]    = sector_percentile(df, "returnOnAssets", True)
    df["s_roic"]   = sector_percentile(df, "roic",           True)
    df["s_pm"]     = sector_percentile(df, "profitMargins",  True)
    df["s_tr_roe"] = sector_percentile(df, "tr_roe",         True)
    df["pillar_profitability"] = df[["s_roe","s_roa","s_roic","s_pm","s_tr_roe"]].mean(axis=1, skipna=True)

    # 3. Growth – v5.3: fcf_to_ni REMOVED (double-count with FCF), replaced with eps_revision_pct_30d
    df["s_rev_g"]        = sector_percentile(df, "revenueGrowth",            True)
    df["s_earn_g"]       = sector_percentile(df, "earningsGrowth",           True)
    df["s_eps_rev_30d"]  = sector_percentile(df, "eps_revision_pct_30d",     True)   # v5.3
    df["s_tr_asset_g"]   = sector_percentile(df, "tr_asset_growth",          False)
    df["s_earn_rev"]     = sector_percentile(df, "earnings_revision_score",  True)
    df["pillar_growth"]  = df[["s_rev_g","s_earn_g","s_eps_rev_30d","s_tr_asset_g","s_earn_rev"]].mean(axis=1, skipna=True)

    # 4-9. שאר הפילרים (ללא שינוי – כבר תואמים v5.3)
    df["s_eq"] = sector_percentile(df, "earnings_quality_score", True)
    df["pillar_earnings_quality"] = df["s_eq"]

    df["s_fcf_yield"] = sector_percentile(df, "fcf_yield",  True)
    df["s_fcf_ni"]    = sector_percentile(df, "fcf_to_ni",  True)
    df["s_fcf_m"]     = sector_percentile(df, "fcf_margin", True)
    df["pillar_fcf"]  = df[["s_fcf_yield","s_fcf_ni","s_fcf_m"]].mean(axis=1, skipna=True)

    df["s_cr"]     = sector_percentile(df, "currentRatio", True)
    df["s_de"]     = sector_percentile(df, "debtToEquity", False)
    df["s_altman"] = sector_percentile(df, "altman_z",     True)
    df["s_beta"]   = sector_percentile(df, "beta",         False)
    df["pillar_health"] = df[["s_cr","s_de","s_altman","s_beta"]].mean(axis=1, skipna=True)

    df["s_mom"]        = sector_percentile(df, "momentum_composite", True)
    df["s_tr_mom12"]   = sector_percentile(df, "tr_momentum_12m",    True)
    df["s_tr_sma"]     = sector_percentile(df, "tr_sma_num",         True)
    df["s_short"]      = sector_percentile(df, "shortRatio",         False)  # v5.3
    df["pillar_momentum"] = df[["s_mom","s_tr_mom12","s_tr_sma","s_short"]].mean(axis=1, skipna=True)

    # Analyst pillar (SmartScore dominant – weight 0 in composite)
    df["s_rec"]          = sector_percentile(df, "recommendationMean",   False)
    df["s_pt_upside"]    = sector_percentile(df, "pt_upside",            True)
    df["s_tr_smart"]     = sector_percentile(df, "tr_smart_score",       True)
    df["s_tr_pt"]        = sector_percentile(df, "tr_pt_upside",         True)
    s_pt_avg = df[["s_pt_upside","s_tr_pt"]].mean(axis=1, skipna=True)
    df["pillar_analyst"] = (
        0.60 * df["s_tr_smart"].fillna(df["s_rec"]) +
        0.25 * s_pt_avg +
        0.15 * df["s_rec"].fillna(50)
    )
    _min, _max = df["pillar_analyst"].min(), df["pillar_analyst"].max()
    if _max > _min:
        df["pillar_analyst"] = ((df["pillar_analyst"] - _min) / (_max - _min)) * 90 + 10

    df["s_piotroski"]      = sector_percentile(df, "piotroski_score", True)
    df["pillar_piotroski"] = df["s_piotroski"]

    df["s_rs_12m"] = sector_percentile(df, "rs_12m", True)
    df["s_rs_6m"]  = sector_percentile(df, "rs_6m",  True)
    df["s_rs_3m"]  = sector_percentile(df, "rs_3m",  True)
    df["pillar_relative_strength"] = df[["s_rs_12m", "s_rs_6m", "s_rs_3m"]].mean(axis=1, skipna=True)

    return df


# ════════════════════════════════════════════════════════════
#  COMPOSITE + FALLBACK (v5.3)
# ════════════════════════════════════════════════════════════
PILLAR_MAP = {
    "valuation":          "pillar_valuation",
    "profitability":      "pillar_profitability",
    "growth":             "pillar_growth",
    "earnings_quality":   "pillar_earnings_quality",
    "fcf_quality":        "pillar_fcf",
    "financial_health":   "pillar_health",
    "momentum":           "pillar_momentum",
    "relative_strength":  "pillar_relative_strength",
    "analyst":            "pillar_analyst",
    "piotroski":          "pillar_piotroski",
}

_TR_AVAILABLE = True   # global flag for frontend banner

def compute_composite(row: pd.Series, weights: dict = None) -> float:
    global _TR_AVAILABLE
    if weights is None:
        weights = CFG["weights"]
    total_w, score = 0.0, 0.0
    for key, col in PILLAR_MAP.items():
        val = row.get(col, np.nan)
        if not pd.isna(val):
            w = weights[key]
            score   += w * val
            total_w += w
    return round(score / total_w, 2) if total_w > 0 else np.nan


# (כל שאר הפונקציות – compute_valuation_score, compute_coverage, add_sector_context,
# caching, Excel export, plots, summary, JSON export – ללא שינוי חוץ משמות קבצים)

# ════════════════════════════════════════════════════════════
#  RUN PIPELINE
# ════════════════════════════════════════════════════════════
def run_pipeline(use_cache: bool = True) -> pd.DataFrame:
    global _TR_AVAILABLE
    # ... (הקוד זהה – רק שינויי שמות קבצים + _TR_AVAILABLE)
    # בסוף:
    style_and_export(df, CFG["output_file"])
    # ...
    print(f"✅  DONE! v5.3 – {CFG['output_file']}")
    return df


if __name__ == "__main__":
    run_pipeline(use_cache=True)
