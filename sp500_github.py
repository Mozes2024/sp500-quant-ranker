# ============================================================
#  S&P 500 ADVANCED RANKING SYSTEM  v5.2 – GitHub Edition
#  Headless · Actions Cache · Parallel Fetch · Full Pipeline
#  Runs as a scheduled GitHub Actions job – no widgets/Colab
#  Outputs: artifacts/sp500_ranking_v5.2.xlsx + 6 PNGs
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


# ════════════════════════════════════════════════════════════
#  TIPRANKS
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


# ════════════════════════════════════════════════════════════
#  S&P 500 TICKERS  (3 fallback strategies)
# ════════════════════════════════════════════════════════════
WIKI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def get_sp500_tickers() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    try:
        resp = requests.get(url, headers=WIKI_HEADERS, timeout=15)
        resp.raise_for_status()
        soup  = BeautifulSoup(resp.text, "html.parser")
        table = (soup.find("table", {"id": "constituents"})
                 or soup.find("table", {"class": "wikitable"}))
        if table is None:
            raise ValueError("Table not found")
        rows = []
        for tr in table.find_all("tr")[1:]:
            cols = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cols) >= 4:
                rows.append({"ticker":   cols[0].replace(".", "-"),
                             "name":     cols[1],
                             "sector":   cols[2],
                             "industry": cols[3]})
        if rows:
            df = pd.DataFrame(rows)
            print(f"✅  {len(df)} tickers (BeautifulSoup)")
            return df
    except Exception as e:
        print(f"  ⚠️  Strategy 1 failed: {e}")

    try:
        tables = pd.read_html(url, attrs={"id": "constituents"}) or pd.read_html(url)
        raw = tables[0]
        raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
        tc = next((c for c in raw.columns if "symbol" in c or "ticker" in c), raw.columns[0])
        nc = next((c for c in raw.columns if "security" in c or "name"   in c), raw.columns[1])
        sc = next((c for c in raw.columns if "sector"  in c),                    raw.columns[2])
        ic = next((c for c in raw.columns if "industry" in c),                   raw.columns[3])
        df = pd.DataFrame({
            "ticker":   raw[tc].astype(str).str.replace(".", "-", regex=False),
            "name":     raw[nc].astype(str),
            "sector":   raw[sc].astype(str),
            "industry": raw[ic].astype(str),
        })
        print(f"✅  {len(df)} tickers (read_html)")
        return df
    except Exception as e:
        print(f"  ⚠️  Strategy 2 failed: {e}")

    try:
        raw = pd.read_csv(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
            "main/data/constituents.csv")
        raw.columns = [c.lower() for c in raw.columns]
        df = pd.DataFrame({
            "ticker":   raw["symbol"].str.replace(".", "-", regex=False),
            "name":     raw.get("name",     raw.get("security",     "")),
            "sector":   raw.get("sector",   "Unknown"),
            "industry": raw.get("sub-industry", raw.get("industry", "Unknown")),
        })
        print(f"✅  {len(df)} tickers (GitHub CSV)")
        return df
    except Exception as e:
        raise RuntimeError(f"All 3 ticker strategies failed: {e}")


# ════════════════════════════════════════════════════════════
#  YAHOO FINANCE  (parallel)
# ════════════════════════════════════════════════════════════
FUNDAMENTAL_FIELDS = [
    "trailingPE", "forwardPE", "pegRatio", "priceToSalesTrailing12Months",
    "priceToBook", "enterpriseToEbitda", "returnOnEquity", "returnOnAssets",
    "profitMargins", "grossMargins", "operatingMargins",
    "revenueGrowth", "earningsGrowth",
    "currentRatio", "debtToEquity", "totalDebt", "totalCash",
    "freeCashflow", "operatingCashflow", "totalRevenue", "netIncomeToCommon",
    "ebitda", "totalAssets", "totalStockholdersEquity", "marketCap",
    "enterpriseValue", "dividendYield", "payoutRatio",
    "beta", "sharesOutstanding", "shortRatio",
    "targetMeanPrice", "currentPrice", "52WeekChange",
    "recommendationMean", "numberOfAnalystOpinions",
    "workingCapital", "earningsPerShare", "trailingEps", "revenuePerShare",
    "averageVolume",
]


def _get_one(ticker: str) -> tuple:
    try:
        info = yf.Ticker(ticker).info
        return ticker, {k: info.get(k) for k in FUNDAMENTAL_FIELDS}
    except Exception:
        return ticker, {}


def fetch_yf_parallel(tickers: list) -> dict:
    results = {}
    with ThreadPoolExecutor(max_workers=CFG["max_workers_yf"]) as executor:
        futures = {executor.submit(_get_one, t): t for t in tickers}
        for future in tqdm(as_completed(futures), total=len(tickers),
                           desc="Yahoo Finance (parallel)"):
            t, info = future.result()
            results[t] = info
    return results


def fetch_price_multi(tickers: list) -> pd.DataFrame:
    try:
        raw = yf.download(tickers, period="2y", auto_adjust=True,
                          progress=False, threads=True)["Close"]
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(tickers[0])
        return raw
    except Exception:
        return pd.DataFrame()


def add_price_momentum(df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    prices = fetch_price_multi(tickers)
    if prices.empty:
        for col in ["perf_12m", "perf_6m", "perf_3m", "perf_1m", "momentum_composite"]:
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

    for col in ["perf_12m", "perf_6m", "perf_3m", "perf_1m", "momentum_composite"]:
        df[col] = df[col].clip(-0.80, 5.0)
    return df


# ════════════════════════════════════════════════════════════
#  COMPUTED METRICS
# ════════════════════════════════════════════════════════════

def _safe(val, default=np.nan):
    if val is None:
        return default
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except Exception:
        return default


def compute_piotroski(row: pd.Series) -> float:
    score = 0
    try:
        ta    = _safe(row.get("totalAssets"), 1)
        roa   = _safe(row.get("returnOnAssets"))
        op_cf = _safe(row.get("operatingCashflow"))
        if roa > 0:                                          score += 1
        if op_cf > 0:                                        score += 1
        if (op_cf / ta) > 0:                                 score += 1
        if _safe(row.get("debtToEquity"), 999) < 50:        score += 1
        if _safe(row.get("currentRatio")) > 1:               score += 1
        if _safe(row.get("grossMargins")) > 0.20:            score += 1
        if (_safe(row.get("totalRevenue"), 0) / ta) > 0.50: score += 1
        if _safe(row.get("revenueGrowth")) > 0:              score += 1
    except Exception:
        pass
    return score


def compute_altman(row: pd.Series) -> float:
    try:
        ta  = _safe(row.get("totalAssets"), 1)
        wc  = _safe(row.get("workingCapital"))
        re_ = _safe(row.get("totalStockholdersEquity"))
        eb  = _safe(row.get("ebitda"))
        mv  = _safe(row.get("marketCap"))
        td  = _safe(row.get("totalDebt"), 1)
        rev = _safe(row.get("totalRevenue"))
        return round(1.2*(wc/ta) + 1.4*(re_/ta) + 3.3*(eb/ta)
                     + 0.6*(mv/td) + 1.0*(rev/ta), 3)
    except Exception:
        return np.nan


def compute_roic(row: pd.Series) -> float:
    try:
        ebitda  = _safe(row.get("ebitda"))
        equity  = _safe(row.get("totalStockholdersEquity"), 0)
        debt    = _safe(row.get("totalDebt"), 0)
        cash    = _safe(row.get("totalCash"), 0)
        nopat   = ebitda * 0.82 * (1 - 0.21)
        inv_cap = equity + debt - cash
        return nopat / inv_cap if inv_cap > 1e6 else np.nan
    except Exception:
        return np.nan


def compute_fcf_metrics(row: pd.Series) -> dict:
    fcf = _safe(row.get("freeCashflow"))
    rev = _safe(row.get("totalRevenue"))
    ni  = _safe(row.get("netIncomeToCommon"))
    mc  = _safe(row.get("marketCap"))
    return {
        "fcf_yield":  fcf / mc  if (fcf and mc  > 0) else np.nan,
        "fcf_margin": fcf / rev if (fcf and rev > 0) else np.nan,
        "fcf_to_ni":  fcf / ni  if (fcf and ni  > 0) else np.nan,
    }


def compute_earnings_quality(row: pd.Series) -> float:
    score = 0
    try:
        ni   = _safe(row.get("netIncomeToCommon"))
        fcf  = _safe(row.get("freeCashflow"))
        ta   = _safe(row.get("totalAssets"), 1)
        roic = _safe(row.get("roic"))
        pm   = _safe(row.get("profitMargins"))
        gm   = _safe(row.get("grossMargins"))
        if not (np.isnan(ni) or np.isnan(fcf)):
            accrual = (ni - fcf) / ta
            if accrual < 0.03:   score += 2
            elif accrual < 0.08: score += 1
        if not np.isnan(roic):
            if roic > 0.15:   score += 1.5
            elif roic > 0.08: score += 0.5
        if not np.isnan(pm) and pm > 0.18: score += 0.5
        if not np.isnan(gm) and gm > 0.40: score += 0.5
    except Exception:
        pass
    return min(score, 5)


def compute_pt_upside(row: pd.Series) -> float:
    t = _safe(row.get("targetMeanPrice"))
    c = _safe(row.get("currentPrice"))
    return (t / c - 1) if (t and c and c > 0) else np.nan


def compute_tr_pt_upside(row: pd.Series) -> float:
    t = _safe(row.get("tr_price_target"))
    c = _safe(row.get("currentPrice"))
    return (t / c - 1) if (t and c and c > 0) else np.nan


# ════════════════════════════════════════════════════════════
#  DYNAMIC SECTOR THRESHOLDS
# ════════════════════════════════════════════════════════════

GLOBAL_THRESHOLDS = {
    "trailingPE":                   (8,    45),
    "forwardPE":                    (8,    40),
    "pegRatio":                     (0.4,   3.5),
    "priceToSalesTrailing12Months": (0.6,   9),
    "priceToBook":                  (0.6,  12),
    "enterpriseToEbitda":           (5,    28),
    "fcf_yield":                    (0.008, 0.09),
    "pt_upside":                    (-0.12, 0.35),
    "tr_pt_upside":                 (-0.12, 0.35),
}

_SECTOR_THRESHOLDS: dict = {}


def build_sector_thresholds(df: pd.DataFrame) -> dict:
    mult_cols = list(GLOBAL_THRESHOLDS.keys())
    thresholds = {}
    for col in mult_cols:
        thresholds[col] = {}
        for sector, grp in df.groupby("sector"):
            vals = grp[col].dropna() if col in grp.columns else pd.Series(dtype=float)
            if len(vals) < 5:
                thresholds[col][sector] = GLOBAL_THRESHOLDS.get(col, (8, 40))
                continue
            med = vals.median()
            std = vals.std()
            thresholds[col][sector] = (
                max(med - 1.5 * std, vals.quantile(0.10)),
                min(med + 1.5 * std, vals.quantile(0.90)),
            )
    return thresholds


# ════════════════════════════════════════════════════════════
#  NORMALISATION
# ════════════════════════════════════════════════════════════

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
    df["s_pe"]        = sector_percentile(df, "trailingPE",                  False)
    df["s_peg"]       = sector_percentile(df, "pegRatio",                    False)
    df["s_ev_ebitda"] = sector_percentile(df, "enterpriseToEbitda",           False)
    df["s_ps"]        = sector_percentile(df, "priceToSalesTrailing12Months", False)
    df["s_pb"]        = sector_percentile(df, "priceToBook",                  False)
    df["pillar_valuation"] = df[["s_pe","s_peg","s_ev_ebitda","s_ps","s_pb"]].mean(axis=1, skipna=True)

    df["s_roe"]    = sector_percentile(df, "returnOnEquity", True)
    df["s_roa"]    = sector_percentile(df, "returnOnAssets", True)
    df["s_roic"]   = sector_percentile(df, "roic",           True)
    df["s_pm"]     = sector_percentile(df, "profitMargins",  True)
    df["s_tr_roe"] = sector_percentile(df, "tr_roe",         True)
    df["pillar_profitability"] = df[["s_roe","s_roa","s_roic","s_pm","s_tr_roe"]].mean(axis=1, skipna=True)

    df["s_rev_g"]      = sector_percentile(df, "revenueGrowth",   True)
    df["s_earn_g"]     = sector_percentile(df, "earningsGrowth",  True)
    df["s_fcf_m"]      = sector_percentile(df, "fcf_margin",      True)
    df["s_tr_asset_g"] = sector_percentile(df, "tr_asset_growth", True)
    df["pillar_growth"] = df[["s_rev_g","s_earn_g","s_fcf_m","s_tr_asset_g"]].mean(axis=1, skipna=True)

    df["s_eq"] = sector_percentile(df, "earnings_quality_score", True)
    df["pillar_earnings_quality"] = df["s_eq"]

    df["s_fcf_yield"] = sector_percentile(df, "fcf_yield",  True)
    df["s_fcf_ni"]    = sector_percentile(df, "fcf_to_ni",  True)
    df["pillar_fcf"]  = df[["s_fcf_yield","s_fcf_ni","s_fcf_m"]].mean(axis=1, skipna=True)

    df["s_cr"]     = sector_percentile(df, "currentRatio", True)
    df["s_de"]     = sector_percentile(df, "debtToEquity", False)
    df["s_altman"] = sector_percentile(df, "altman_z",     True)
    df["pillar_health"] = df[["s_cr","s_de","s_altman"]].mean(axis=1, skipna=True)

    df["s_mom"]      = sector_percentile(df, "momentum_composite", True)
    df["s_beta"]     = sector_percentile(df, "beta",               False)
    df["s_tr_mom12"] = sector_percentile(df, "tr_momentum_12m",    True)
    df["s_tr_sma"]   = sector_percentile(df, "tr_sma_num",         True)
    df["pillar_momentum"] = df[["s_mom","s_beta","s_tr_mom12","s_tr_sma"]].mean(axis=1, skipna=True)

    df["s_rec"]          = sector_percentile(df, "recommendationMean",   False)
    df["s_pt_upside"]    = sector_percentile(df, "pt_upside",            True)
    df["s_tr_smart"]     = sector_percentile(df, "tr_smart_score",       True)
    df["s_tr_consensus"] = sector_percentile(df, "tr_consensus_num",     True)
    df["s_tr_news"]      = sector_percentile(df, "tr_news_sent_num",     True)
    df["s_tr_blogger"]   = sector_percentile(df, "tr_blogger_bullish",   True)
    df["s_tr_hedge"]     = sector_percentile(df, "tr_hedge_trend_num",   True)
    df["s_tr_insider"]   = sector_percentile(df, "tr_insider_trend_num", True)
    df["s_tr_inv_chg"]   = sector_percentile(df, "tr_investor_chg_30d",  True)
    df["s_tr_pt"]        = sector_percentile(df, "tr_pt_upside",         True)
    df["pillar_analyst"] = df[[
        "s_rec","s_pt_upside","s_tr_smart","s_tr_consensus","s_tr_news",
        "s_tr_blogger","s_tr_hedge","s_tr_insider","s_tr_inv_chg","s_tr_pt",
    ]].mean(axis=1, skipna=True)

    df["s_piotroski"]      = sector_percentile(df, "piotroski_score", True)
    df["pillar_piotroski"] = df["s_piotroski"]

    return df


# ════════════════════════════════════════════════════════════
#  COMPOSITE SCORE
# ════════════════════════════════════════════════════════════

PILLAR_MAP = {
    "valuation":        "pillar_valuation",
    "profitability":    "pillar_profitability",
    "growth":           "pillar_growth",
    "earnings_quality": "pillar_earnings_quality",
    "fcf_quality":      "pillar_fcf",
    "financial_health": "pillar_health",
    "momentum":         "pillar_momentum",
    "analyst":          "pillar_analyst",
    "piotroski":        "pillar_piotroski",
}


def compute_composite(row: pd.Series, weights: dict = None) -> float:
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


# ════════════════════════════════════════════════════════════
#  VALUATION SCORE  1–100
# ════════════════════════════════════════════════════════════

def compute_valuation_score(row: pd.Series) -> float:
    signals = []
    sector  = row.get("sector", "__global__")

    def add(col, invert=False):
        raw = _safe(row.get(col))
        if pd.isna(raw):
            return
        cheap_thr, exp_thr = (_SECTOR_THRESHOLDS.get(col, {})
                               .get(sector, GLOBAL_THRESHOLDS.get(col, (0, 1))))
        rng  = max(exp_thr - cheap_thr, 1e-9)
        norm = (raw - cheap_thr) / rng if invert else (exp_thr - raw) / rng
        signals.append(float(np.clip(norm, 0, 1)) * 99 + 1)

    add("trailingPE")
    add("forwardPE")
    add("pegRatio")
    add("priceToSalesTrailing12Months")
    add("priceToBook")
    add("enterpriseToEbitda")
    add("fcf_yield", invert=True)
    if not pd.isna(row.get("tr_pt_upside", np.nan)):
        add("tr_pt_upside", invert=True)
    else:
        add("pt_upside", invert=True)

    return round(float(np.clip(np.mean(signals), 1, 100)), 1) if signals else np.nan


# ════════════════════════════════════════════════════════════
#  COVERAGE + SECTOR CONTEXT
# ════════════════════════════════════════════════════════════

CORE_METRIC_COLS = [
    "trailingPE", "returnOnEquity", "returnOnAssets", "profitMargins",
    "revenueGrowth", "earningsGrowth", "currentRatio", "debtToEquity",
    "freeCashflow", "altman_z", "piotroski_score", "beta",
    "recommendationMean", "fcf_yield", "tr_smart_score",
    "earnings_quality_score", "momentum_composite",
]


def compute_coverage(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda r: _coverage(r, CORE_METRIC_COLS), axis=1)


def _coverage(row: pd.Series, cols: list) -> float:
    return sum(1 for c in cols if not pd.isna(row.get(c, np.nan))) / max(len(cols), 1)


def add_sector_context(df: pd.DataFrame) -> pd.DataFrame:
    pillar_cols = list(PILLAR_MAP.values())
    sector_med  = (df.groupby("sector")[pillar_cols + ["composite_score"]]
                   .median().add_prefix("sector_med_"))
    df = df.merge(sector_med, on="sector", how="left")
    df["vs_sector"] = df["composite_score"] - df["sector_med_composite_score"]
    return df


# ════════════════════════════════════════════════════════════
#  CACHING
# ════════════════════════════════════════════════════════════

def load_cache() -> "pd.DataFrame | None":
    global _SECTOR_THRESHOLDS
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "rb") as f:
            payload = pickle.load(f)
        if len(payload) == 3:
            data, saved_thresholds, ts = payload
            _SECTOR_THRESHOLDS = saved_thresholds
        else:
            data, ts = payload
            _SECTOR_THRESHOLDS = build_sector_thresholds(data)
            try:
                with open(CACHE_FILE, "wb") as fw:
                    pickle.dump((data, _SECTOR_THRESHOLDS, ts), fw)
            except Exception:
                pass
        age = datetime.now() - ts
        if age < timedelta(hours=CFG["cache_hours"]):
            print(f"✅  Cache loaded ({int(age.total_seconds()//60)} min old)")
            return data
        print(f"  ℹ️  Cache expired ({int(age.total_seconds()//3600)}h old) — refreshing")
    except Exception as e:
        print(f"  ⚠️  Cache read error: {e}")
    return None


def save_cache(df: pd.DataFrame):
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump((df, _SECTOR_THRESHOLDS, datetime.now()), f)
        print(f"💾  Cache saved → {CACHE_FILE}")
    except Exception as e:
        print(f"  ⚠️  Cache save error: {e}")


# ════════════════════════════════════════════════════════════
#  EXCEL EXPORT
# ════════════════════════════════════════════════════════════

EXPORT_COLS = [
    "rank", "ticker", "name", "sector", "industry",
    "composite_score", "valuation_score",
    "pillar_valuation", "pillar_profitability", "pillar_growth",
    "pillar_earnings_quality", "pillar_fcf", "pillar_health",
    "pillar_momentum", "pillar_analyst", "pillar_piotroski",
    "coverage",
    "tr_smart_score", "tr_analyst_consensus", "tr_consensus_num",
    "tr_news_sentiment", "tr_news_bullish",
    "tr_blogger_bullish", "tr_hedge_trend", "tr_hedge_trend_num",
    "tr_insider_trend", "tr_insider_3m_usd",
    "tr_investor_chg_30d", "tr_investor_chg_7d",
    "tr_momentum_12m", "tr_sma",
    "tr_price_target", "tr_pt_upside", "tr_roe", "tr_asset_growth",
    "trailingPE", "forwardPE", "pegRatio",
    "priceToSalesTrailing12Months", "priceToBook", "enterpriseToEbitda",
    "returnOnEquity", "returnOnAssets", "roic",
    "profitMargins", "grossMargins", "operatingMargins",
    "revenueGrowth", "earningsGrowth",
    "fcf_yield", "fcf_margin", "fcf_to_ni",
    "earnings_quality_score",
    "currentRatio", "debtToEquity",
    "dividendYield", "payoutRatio", "beta",
    "perf_12m", "perf_6m", "perf_3m", "perf_1m", "momentum_composite",
    "altman_z", "piotroski_score",
    "recommendationMean", "numberOfAnalystOpinions", "pt_upside",
    "marketCap", "enterpriseValue", "currentPrice", "averageVolume",
    "vs_sector",
]

FRIENDLY_NAMES = { ... }  # (השאר כמו שהיה)

PCT_COLS_DECIMAL = { ... }
PCT_COLS_FRACTION = { ... }
ALL_PCT_COLS = PCT_COLS_DECIMAL | PCT_COLS_FRACTION

def style_and_export(df: pd.DataFrame, filepath: str):
    # (הקוד המלא כמו שהיה)
    ...

def _format_sheet(ws):
    # (הקוד המלא כמו שהיה)
    ...

# ════════════════════════════════════════════════════════════
#  VISUALISATIONS
# ════════════════════════════════════════════════════════════

def plot_all(df: pd.DataFrame):
    # (הקוד המלא כמו שהיה)
    ...

def _plot_radar(df_top: pd.DataFrame):
    # (הקוד המלא כמו שהיה)
    ...

# ════════════════════════════════════════════════════════════
#  SUMMARY PRINT
# ════════════════════════════════════════════════════════════

def _print_summary(df: pd.DataFrame):
    # (הקוד המלא כמו שהיה)
    ...

# ════════════════════════════════════════════════════════════
#  MERGE BREAKOUT SIGNALS (הגרסה שעבדה)
# ════════════════════════════════════════════════════════════
def merge_breakout_signals(df: pd.DataFrame) -> pd.DataFrame:
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
    print(f"  🔀 Overlap with S&P 500: {n_overlap} stocks")
    return df


# ════════════════════════════════════════════════════════════
#  JSON EXPORT
# ════════════════════════════════════════════════════════════
def export_json(df: pd.DataFrame):
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
            # Breakout
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

    payload = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        "count":     len(records),
        "data":      records,
    }
    json_path = "artifacts/sp500_data.json"
    with open(json_path, "w") as f:
        import json
        json.dump(payload, f, separators=(",", ":"))
    with open("sp500_data.json", "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"✅  JSON → {json_path}  ({len(records)} stocks)")


# ════════════════════════════════════════════════════════════
#  RUN PIPELINE
# ════════════════════════════════════════════════════════════
def run_pipeline(use_cache: bool = True) -> pd.DataFrame:
    global _SECTOR_THRESHOLDS

    print("=" * 65)
    print(f"  S&P 500 ADVANCED RANKING v5.2 – GitHub Edition")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    cached = load_cache() if use_cache else None
    if cached is not None:
        df = cached
        print("  Skipping fetch – using cached data")
    else:
        universe = get_sp500_tickers()
        tickers  = universe["ticker"].tolist()

        print(f"\n[1/6]  Yahoo Finance ({len(tickers)} tickers, parallel)...")
        yf_data = fetch_yf_parallel(tickers)
        fund_df = pd.DataFrame.from_dict(yf_data, orient="index")
        fund_df.index.name = "ticker"
        fund_df = fund_df.reset_index()
        df = universe.merge(fund_df, on="ticker", how="left")

        print("\n[2/6]  TipRanks...")
        tr_df = fetch_tipranks(tickers)
        if not tr_df.empty and "ticker" in tr_df.columns:
            df = df.merge(tr_df, on="ticker", how="left")
        else:
            _TR_COLS = list(_parse_tipranks({}).keys())
            for col in _TR_COLS:
                df[col] = np.nan

        print("\n[3/6]  Computing metrics...")
        df["piotroski_score"]        = df.apply(compute_piotroski,        axis=1)
        df["altman_z"]               = df.apply(compute_altman,           axis=1)
        df["roic"]                   = df.apply(compute_roic,             axis=1)
        fcf_cols = df.apply(compute_fcf_metrics, axis=1, result_type="expand")
        df = pd.concat([df, fcf_cols], axis=1)
        df["pt_upside"]              = df.apply(compute_pt_upside,        axis=1)
        df["tr_pt_upside"]           = df.apply(compute_tr_pt_upside,     axis=1)
        df["earnings_quality_score"] = df.apply(compute_earnings_quality, axis=1)

        df["returnOnEquity"] = df["returnOnEquity"].clip(-2.0, 5.0)
        df["fcf_yield"]      = df["fcf_yield"].clip(-0.50, 0.50)
        df["roic"]           = df["roic"].clip(-1.0, 3.0)

        print("\n[4/6]  Multi-timeframe momentum...")
        df = add_price_momentum(df, tickers)

        before = len(df)
        df = df[
            (df["marketCap"].fillna(0)     >= CFG["min_market_cap"]) &
            (df["averageVolume"].fillna(0) >= CFG["min_avg_volume"])
        ].copy()
        print(f"\n  Liquidity filter: {before} → {len(df)}")

        df["coverage"] = compute_coverage(df)
        before2 = len(df)
        df = df[df["coverage"] >= CFG["min_coverage"]].copy()
        print(f"  Coverage filter:  {before2} → {len(df)}")

        print("\n[5/6]  Dynamic sector thresholds...")
        _SECTOR_THRESHOLDS = build_sector_thresholds(df)

        print("\n[6/6]  Pillar scores...")
        df = build_pillar_scores(df)

        df["composite_score"] = df.apply(compute_composite,       axis=1)
        df["valuation_score"] = df.apply(compute_valuation_score, axis=1)

        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        df = add_sector_context(df)

        save_cache(df)

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
    print("    JSON   → sp500_data.json (with breakout signals)")
    return df


# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_pipeline(use_cache=True)
