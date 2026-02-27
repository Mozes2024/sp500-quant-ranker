# ranker/data_tipranks.py — TipRanks data fetching
import time
import numpy as np
import pandas as pd
import requests
from ranker.config import CFG, TR_URL, TR_HEADERS, _CONSENSUS, _TREND, _SENTIMENT, _SMA

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
