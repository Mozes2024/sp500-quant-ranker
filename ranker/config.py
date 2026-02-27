# ranker/config.py — Configuration, weights, constants, shared state
import numpy as np
import pandas as pd

# ════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════
CFG = {
    "weights": {
        # ── v5.3 Model Enhancement ──────────────────────────────────
        # KEY CHANGES:
        #  • Earnings Revisions promoted to independent pillar (was buried in Growth)
        #    Research: Zacks 1979, Chan/Jegadeesh/Lakonishok 1996 — strongest short-term alpha factor
        #  • Analyst revived at 5% — now includes insider_pct_mcap signal (not just SmartScore)
        #  • Piotroski removed entirely (proxy-based F-Score without Y/Y data = noise)
        #  • Growth trimmed — Earnings Revisions signals extracted to own pillar
        "valuation":          0.14,   # was 0.15 — gave 1% to revived Analyst
        "profitability":      0.18,   # was 0.19 — gave 1% to revived Analyst
        "growth":             0.10,   # trimmed: earn_rev signals moved to own pillar
        "earnings_revisions": 0.10,   # NEW pillar: extracted from Growth (Zacks alpha factor)
        "earnings_quality":   0.10,
        "fcf_quality":        0.12,
        "financial_health":   0.09,
        "momentum":           0.08,
        "relative_strength":  0.07,
        "analyst":            0.02,   # Revived: clean composition — PT Upside + Insider + Yahoo (no SmartScore)
    },
    "min_coverage":    0.45,
    "min_market_cap":  5_000_000_000,
    "min_avg_volume":  500_000,
    "cache_hours":     24,
    "sleep_tr":        0.35,
    "batch_size_tr":   10,
    "max_workers_yf":  20,
    "output_file":     "artifacts/sp500_ranking_v5.3.xlsx",
}
assert abs(sum(CFG["weights"].values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

CACHE_FILE = "sp500_cache_v10.pkl"  # bumped v9→v10: earnings revisions (eps_revisions + eps_trend from Yahoo)

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

WIKI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

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
    "workingCapital", "currentAssets", "currentLiabilities",
    "earningsPerShare", "trailingEps", "revenuePerShare",
    "averageVolume",
    "operatingIncome",  # FIX B1: needed for accurate ROIC (EBIT-based NOPAT)
]

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

PILLAR_MAP = {
    "valuation":           "pillar_valuation",
    "profitability":       "pillar_profitability",
    "growth":              "pillar_growth",
    "earnings_revisions":  "pillar_earnings_revisions",    # NEW: independent pillar
    "earnings_quality":    "pillar_earnings_quality",
    "fcf_quality":         "pillar_fcf",
    "financial_health":    "pillar_health",
    "momentum":            "pillar_momentum",
    "relative_strength":   "pillar_relative_strength",
    "analyst":             "pillar_analyst",
    # Piotroski REMOVED from PILLAR_MAP — still computed for display, but no longer a scoring pillar
}

# FIX v5.3: Fallback weights when TipRanks data is unavailable
# Redistributes TR-dependent weight to pure-fundamental pillars
CFG_WEIGHTS_NO_TR = {
    "valuation":           0.18,
    "profitability":       0.22,
    "growth":              0.12,
    "earnings_revisions":  0.12,   # fully Yahoo-based — no TR dependency
    "earnings_quality":    0.12,
    "fcf_quality":         0.14,
    "financial_health":    0.10,
    "momentum":            0.00,   # can't use TR momentum without TR
    "relative_strength":   0.00,   # keep low without TR confirmation
    "analyst":             0.00,
}

# Track TR availability globally
_TR_AVAILABLE = True

CORE_METRIC_COLS = [
    "trailingPE", "returnOnEquity", "returnOnAssets", "profitMargins",
    "revenueGrowth", "earningsGrowth", "currentRatio", "debtToEquity",
    "freeCashflow", "altman_z", "piotroski_score", "beta",
    "recommendationMean", "fcf_yield", "tr_smart_score",
    "earnings_quality_score", "momentum_composite",
    "earnings_revision_score", "shortRatio", "ma_regime_score",
]

EXPORT_COLS = [
    "rank", "ticker", "name", "sector", "industry",
    "composite_score", "composite_raw", "valuation_score",
    "pillar_valuation", "pillar_profitability", "pillar_growth",
    "pillar_earnings_revisions", "pillar_earnings_quality", "pillar_fcf", "pillar_health",
    "pillar_momentum", "pillar_relative_strength", "pillar_analyst", "pillar_piotroski",
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
    "earnings_revision_score", "rev_ratio_30d", "eps_revision_pct_90d",
    "currentRatio", "debtToEquity",
    "dividendYield", "payoutRatio", "beta",
    "perf_12m", "perf_6m", "perf_3m", "perf_1m", "momentum_composite",
    "rs_12m", "rs_6m", "rs_3m", "rs_1m", "rs_composite",
    "altman_z", "piotroski_score",
    "recommendationMean", "numberOfAnalystOpinions", "pt_upside",
    "marketCap", "enterpriseValue", "currentPrice", "averageVolume",
    "shortRatio", "insider_pct_mcap",
    "ma_regime_score", "pct_above_ma200",
    "vs_sector",
]

FRIENDLY_NAMES = {
    "rank": "Rank", "ticker": "Ticker", "name": "Company",
    "sector": "Sector", "industry": "Industry",
    "composite_score":         "Composite Score",
    "composite_raw":           "Composite (raw, unweighted)",
    "valuation_score":         "Cheap/Expensive (1-100)",
    "pillar_valuation":        "Valuation",
    "pillar_profitability":    "Profitability",
    "pillar_growth":           "Growth",
    "pillar_earnings_revisions": "Earnings Revisions",
    "pillar_earnings_quality": "Earnings Quality",
    "pillar_fcf":              "FCF Quality",
    "pillar_health":           "Financial Health",
    "pillar_momentum":         "Momentum",
    "pillar_relative_strength": "Rel. Strength vs Market",
    "pillar_analyst":          "Analyst+Sentiment",
    "pillar_piotroski":        "Piotroski",
    "coverage":                "Data Coverage %",
    "tr_smart_score":          "TR SmartScore",
    "tr_analyst_consensus":    "TR Consensus Label",
    "tr_consensus_num":        "TR Consensus (0-5)",
    "tr_news_sentiment":       "TR News Sentiment",
    "tr_news_bullish":         "TR News Bullish %",
    "tr_blogger_bullish":      "TR Blogger Bullish %",
    "tr_hedge_trend":          "TR Hedge Fund Trend",
    "tr_hedge_trend_num":      "TR Hedge Num",
    "tr_insider_trend":        "TR Insider Trend",
    "tr_insider_3m_usd":       "TR Insider 3M ($)",
    "tr_investor_chg_30d":     "TR Investor Chg 30D %",
    "tr_investor_chg_7d":      "TR Investor Chg 7D %",
    "tr_momentum_12m":         "TR Tech Mom 12M %",
    "tr_sma":                  "TR SMA Signal",
    "tr_price_target":         "TR Price Target ($)",
    "tr_pt_upside":            "TR PT Upside %",
    "tr_roe":                  "TR ROE %",
    "tr_asset_growth":         "TR Asset Growth %",
    "trailingPE":              "P/E (TTM)",
    "forwardPE":               "Forward P/E",
    "pegRatio":                "PEG",
    "priceToSalesTrailing12Months": "P/S",
    "priceToBook":             "P/B",
    "enterpriseToEbitda":      "EV/EBITDA",
    "returnOnEquity":          "ROE %",
    "returnOnAssets":          "ROA %",
    "roic":                    "ROIC %",
    "profitMargins":           "Net Margin %",
    "grossMargins":            "Gross Margin %",
    "operatingMargins":        "Op. Margin %",
    "revenueGrowth":           "Rev Growth %",
    "earningsGrowth":          "EPS Growth %",
    "fcf_yield":               "FCF Yield %",
    "fcf_margin":              "FCF Margin %",
    "fcf_to_ni":               "FCF/Net Income",
    "earnings_quality_score":  "Earnings Quality (0-5)",
    "earnings_revision_score": "Earnings Revision (0-5)",
    "rev_ratio_30d":           "Rev Ratio 30D (-1 to +1)",
    "eps_revision_pct_90d":    "EPS Est. Chg 90D %",
    "currentRatio":            "Current Ratio",
    "debtToEquity":            "Debt/Equity",
    "dividendYield":           "Div Yield %",
    "payoutRatio":             "Payout Ratio %",
    "beta":                    "Beta",
    "perf_12m":                "Perf 12M %",
    "perf_6m":                 "Perf 6M %",
    "perf_3m":                 "Perf 3M %",
    "perf_1m":                 "Perf 1M %",
    "momentum_composite":      "Momentum Composite %",
    "rs_12m":                  "RS vs SPY 12M %",
    "rs_6m":                   "RS vs SPY 6M %",
    "rs_3m":                   "RS vs SPY 3M %",
    "rs_1m":                   "RS vs SPY 1M %",
    "rs_composite":            "RS Composite %",
    "altman_z":                "Altman Z",
    "piotroski_score":         "Piotroski F",
    "recommendationMean":      "Yahoo Analyst (1=Buy)",
    "numberOfAnalystOpinions": "# Analysts",
    "pt_upside":               "Yahoo PT Upside %",
    "marketCap":               "Market Cap ($)",
    "enterpriseValue":         "EV ($)",
    "currentPrice":            "Price ($)",
    "averageVolume":           "Avg Volume",
    "shortRatio":              "Short Ratio (Days)",
    "insider_pct_mcap":        "Insider Buy/Sell % MCap",
    "ma_regime_score":         "MA Regime (0-2.5)",
    "pct_above_ma200":         "% Above MA200",
    "vs_sector":               "vs Sector Median",
}

PCT_COLS_DECIMAL = {
    "returnOnEquity", "returnOnAssets", "roic",
    "profitMargins", "grossMargins", "operatingMargins",
    "revenueGrowth", "earningsGrowth",
    "fcf_yield", "fcf_margin",
    "dividendYield", "payoutRatio",
    "pt_upside", "coverage",
}

PCT_COLS_FRACTION = {
    "tr_news_bullish", "tr_blogger_bullish",
    "tr_investor_chg_30d", "tr_investor_chg_7d",
    "tr_momentum_12m", "tr_roe", "tr_asset_growth", "tr_pt_upside",
    "perf_12m", "perf_6m", "perf_3m", "perf_1m", "momentum_composite",
    "rs_12m", "rs_6m", "rs_3m", "rs_1m", "rs_composite",
    "eps_revision_pct_90d", "pct_above_ma200",
}

ALL_PCT_COLS = PCT_COLS_DECIMAL | PCT_COLS_FRACTION
