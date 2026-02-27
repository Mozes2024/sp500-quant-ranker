# ranker/composite.py — Composite scoring, valuation, coverage
import numpy as np
import pandas as pd
from ranker.config import CFG, CFG_WEIGHTS_NO_TR, PILLAR_MAP, _TR_AVAILABLE, CORE_METRIC_COLS
from ranker.utils import _coverage

def compute_composite(row: pd.Series, weights: dict = None) -> float:
    if weights is None:
        weights = CFG_WEIGHTS_NO_TR if not _TR_AVAILABLE else CFG["weights"]
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
    "earnings_revision_score", "shortRatio", "ma_regime_score",
]


def compute_coverage(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda r: _coverage(r, CORE_METRIC_COLS), axis=1)


def add_sector_context(df: pd.DataFrame) -> pd.DataFrame:
    pillar_cols = [c for c in PILLAR_MAP.values() if c in df.columns]
    cols = pillar_cols + ["composite_score"]
    cols = [c for c in cols if c in df.columns]
    sector_med  = (df.groupby("sector")[cols]
                   .median().add_prefix("sector_med_"))
    df = df.merge(sector_med, on="sector", how="left")
    df["vs_sector"] = df["composite_score"] - df["sector_med_composite_score"]
    return df
