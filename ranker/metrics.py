# ranker/metrics.py — Computed financial metrics
import numpy as np
import pandas as pd
from ranker.utils import _safe

def compute_piotroski(row: pd.Series) -> float:
    """
    Full 9-point Piotroski F-Score (Piotroski 2000).
    Where prior-year data is unavailable, we use directional proxies from Yahoo Finance.

    PROFITABILITY (F1–F4):
      F1 ROA > 0                 — direct
      F2 OCF > 0                 — direct
      F3 delta ROA > 0           — proxy: earningsGrowth > 0
      F4 Accruals: OCF/TA > ROA  — direct (earnings quality signal)

    LEVERAGE / LIQUIDITY (F5–F7):
      F5 delta Leverage < 0      — proxy: D/E < 1.0 (tighter than old threshold of 50)
      F6 delta Current Ratio > 0 — proxy: CR > 1.5 (tighter than old threshold of 1.0)
      F7 No share dilution       — proxy: sharesOutstanding reasonable (skip if unavailable)

    OPERATING EFFICIENCY (F8–F9):
      F8 delta Gross Margin > 0  — proxy: grossMargins > 0.25
      F9 delta Asset Turnover > 0 — proxy: (revenue/assets > 0.4) AND revenueGrowth > 0
    """
    score = 0
    try:
        ta    = _safe(row.get("totalAssets"), 1)
        roa   = _safe(row.get("returnOnAssets"))
        op_cf = _safe(row.get("operatingCashflow"))

        # F1 — ROA positive
        if not np.isnan(roa) and roa > 0:
            score += 1

        # F2 — Operating cash flow positive
        if not np.isnan(op_cf) and op_cf > 0:
            score += 1

        # F3 — delta ROA proxy: earnings grew (earningsGrowth > 0)
        eg = _safe(row.get("earningsGrowth"))
        if not np.isnan(eg) and eg > 0:
            score += 1

        # F4 — Accruals: cash earnings > accounting earnings (quality signal)
        # OCF/TA > ROA means real cash exceeds reported profit rate
        if not np.isnan(op_cf) and not np.isnan(roa):
            if (op_cf / ta) > roa:
                score += 1

        # F5 — Leverage: D/E below 1.0 (conservative threshold, replaces old D/E < 50)
        de = _safe(row.get("debtToEquity"), 999)
        if de < 100:   # D/E reported as %, so 100 = 1.0x
            score += 1

        # F6 — Liquidity: current ratio > 1.5 (tighter than old > 1.0)
        cr = _safe(row.get("currentRatio"))
        if not np.isnan(cr) and cr > 1.5:
            score += 1

        # F7 — No dilution proxy: payout ratio reasonable (company not funding itself via equity)
        # Skip if data unavailable (don't penalise for missing data)
        pr = _safe(row.get("payoutRatio"))
        shares = _safe(row.get("sharesOutstanding"))
        mc     = _safe(row.get("marketCap"))
        price  = _safe(row.get("currentPrice"))
        if not (np.isnan(shares) or np.isnan(mc) or np.isnan(price)) and price > 0:
            implied_shares = mc / price
            # If reported shares within 5% of implied → no significant dilution
            if abs(shares - implied_shares) / max(implied_shares, 1) < 0.05:
                score += 1
        elif not np.isnan(pr) and pr < 1.0:
            score += 1   # fallback: sustainable payout suggests no equity pressure

        # F8 — delta Gross Margin proxy: gross margin > 25%
        gm = _safe(row.get("grossMargins"))
        if not np.isnan(gm) and gm > 0.25:
            score += 1

        # F9 — delta Asset Turnover proxy: asset turnover > 0.4 AND revenue growing
        rev = _safe(row.get("totalRevenue"), 0)
        rg  = _safe(row.get("revenueGrowth"))
        if (rev / ta) > 0.40 and not np.isnan(rg) and rg > 0:
            score += 1

    except Exception:
        pass
    return score


# Sectors where Altman Z is not meaningful (financials have debt as product, REITs use different metrics)
_ALTMAN_SKIP_SECTORS = {"Financials", "Real Estate"}

# Manufacturing sectors → use original Altman Z (includes Sales/TA)
_ALTMAN_MFG_SECTORS  = {"Industrials", "Energy", "Materials", "Consumer Discretionary",
                         "Consumer Staples"}

# Non-manufacturing / service sectors → use Altman Z'' (no Sales/TA ratio)
# Thresholds: Z'' > 2.6 = safe, 1.1–2.6 = grey, < 1.1 = distress
_ALTMAN_SVC_SECTORS  = {"Information Technology", "Health Care",
                         "Communication Services", "Utilities"}

# FIX B3: Explicit set of ALL known S&P 500 GICS sectors for safety validation
_ALL_KNOWN_SECTORS = _ALTMAN_SKIP_SECTORS | _ALTMAN_MFG_SECTORS | _ALTMAN_SVC_SECTORS
_altman_unknown_sectors_warned = set()  # track already-warned sectors


def compute_altman(row: pd.Series) -> float:
    """
    Sector-aware Altman Z-Score (3 models):

    SKIP  — Financials, Real Estate: model not valid (debt is the product)

    Z     — Manufacturing (Industrials, Energy, Materials, Consumer):
            1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
            Thresholds: > 3.0 safe | 1.8–3.0 grey | < 1.8 distress

    Z\'\'  — Non-manufacturing / Services (Tech, Health, Comm, Utilities):
            6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
            No Sales/TA ratio — designed for asset-light companies
            Thresholds: > 2.6 safe | 1.1–2.6 grey | < 1.1 distress

    X1 = Working Capital / Total Assets
    X2 = Retained Earnings (Equity) / Total Assets
    X3 = EBITDA / Total Assets  (proxy for EBIT)
    X4 = Market Value Equity / Total Debt
    X5 = Revenue / Total Assets  (Z model only)
    """
    try:
        sector = row.get("sector", "") or ""

        # ── Skip financials & REITs ──────────────────────────────────────
        if sector in _ALTMAN_SKIP_SECTORS:
            return np.nan

        # ── Shared inputs ────────────────────────────────────────────────
        ta = _safe(row.get("totalAssets"), 1)

        wc = _safe(row.get("workingCapital"))
        if np.isnan(wc):
            ca = _safe(row.get("currentAssets"))
            cl = _safe(row.get("currentLiabilities"))
            if not (np.isnan(ca) or np.isnan(cl)):
                wc = ca - cl
            else:
                return np.nan

        re_ = _safe(row.get("totalStockholdersEquity"))   # retained earnings proxy
        eb  = _safe(row.get("ebitda"))
        mv  = _safe(row.get("marketCap"))
        td  = _safe(row.get("totalDebt"), 1)

        if any(np.isnan(v) for v in [wc, re_, eb, mv]):
            return np.nan

        x1 = wc  / ta
        x2 = re_ / ta
        x3 = eb  / ta
        x4 = mv  / max(td, 1)

        # ── Z model: manufacturing / asset-heavy sectors ─────────────────
        if sector in _ALTMAN_MFG_SECTORS:
            rev = _safe(row.get("totalRevenue"))
            if np.isnan(rev):
                return np.nan
            x5 = rev / ta
            return round(1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5, 3)

        # ── FIX B3: Warn if sector is unknown (not in any known set) ────
        if sector and sector not in _ALL_KNOWN_SECTORS and sector not in _altman_unknown_sectors_warned:
            _altman_unknown_sectors_warned.add(sector)
            print(f"  ⚠️  Altman Z: unknown sector '{sector}' — using Z'' (service) model as fallback. "
                  f"Add to _ALTMAN_MFG_SECTORS or _ALTMAN_SVC_SECTORS for correct model.")

        # ── Z'' model: service / asset-light sectors (default) ───────────
        return round(6.56*x1 + 3.26*x2 + 6.72*x3 + 1.05*x4, 3)

    except Exception:
        return np.nan


def compute_roic(row: pd.Series) -> float:
    """
    Return on Invested Capital = NOPAT / Invested Capital

    FIX B1: The old formula used EBITDA * 0.82 * (1 - tax) as NOPAT proxy,
    where 0.82 was a fixed D&A-to-EBITDA ratio. This penalized asset-light
    companies (Tech D&A ~5% of EBITDA) and rewarded asset-heavy ones
    (Industrials D&A ~25%).

    New approach (priority order):
    1. operatingIncome * (1 - tax) — best available proxy if Yahoo provides it
    2. EBITDA * sector_da_ratio * (1 - tax) — sector-aware fallback
    3. EBITDA * 0.82 * (1 - tax) — legacy fallback if sector unknown
    """
    TAX_RATE = 0.21
    # Sector-specific D&A-to-EBITDA retention ratios (1 - D&A/EBITDA)
    # Source: Damodaran sector averages, rounded for stability
    _SECTOR_DA_RATIO = {
        "Information Technology":    0.90,   # low D&A — asset-light
        "Communication Services":    0.80,   # moderate (media assets)
        "Health Care":               0.85,   # moderate (R&D-heavy but capitalized)
        "Financials":                0.95,   # very low D&A
        "Consumer Discretionary":    0.78,   # moderate-high (retail, auto)
        "Consumer Staples":          0.80,   # moderate
        "Industrials":               0.75,   # high D&A — asset-heavy
        "Energy":                    0.65,   # very high D&A (PP&E intensive)
        "Materials":                 0.70,   # high D&A
        "Real Estate":               0.60,   # very high D&A (depreciation of properties)
        "Utilities":                 0.65,   # very high D&A (infrastructure)
    }
    try:
        equity  = _safe(row.get("totalStockholdersEquity"), 0)
        debt    = _safe(row.get("totalDebt"), 0)
        cash    = _safe(row.get("totalCash"), 0)
        inv_cap = equity + debt - cash
        if inv_cap <= 1e6:
            return np.nan

        # Priority 1: Operating Income (EBIT) if available
        op_income = _safe(row.get("operatingIncome"))
        if not np.isnan(op_income):
            nopat = op_income * (1 - TAX_RATE)
            return nopat / inv_cap

        # Priority 2: EBITDA with sector-aware D&A ratio
        ebitda = _safe(row.get("ebitda"))
        if np.isnan(ebitda):
            return np.nan
        sector = row.get("sector", "") or ""
        da_ratio = _SECTOR_DA_RATIO.get(sector, 0.82)  # legacy fallback
        nopat = ebitda * da_ratio * (1 - TAX_RATE)
        return nopat / inv_cap
    except Exception:
        return np.nan


def compute_fcf_metrics(row: pd.Series) -> dict:
    fcf = _safe(row.get("freeCashflow"))
    rev = _safe(row.get("totalRevenue"))
    ni  = _safe(row.get("netIncomeToCommon"))
    mc  = _safe(row.get("marketCap"))
    # FIX B2: Python truthy bug — `(0 and x)` = 0 (falsy), so FCF=0 was silently
    # mapped to NaN instead of 0.0. Use explicit NaN checks + denominator > 0.
    def _ratio(numerator, denominator):
        if np.isnan(numerator) or np.isnan(denominator) or denominator <= 0:
            return np.nan
        return numerator / denominator
    return {
        "fcf_yield":  _ratio(fcf, mc),
        "fcf_margin": _ratio(fcf, rev),
        "fcf_to_ni":  _ratio(fcf, ni),
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
            if accrual < 0:      score += 2   # FIX: FCF > NI = real cash exceeds accounting profit (strong signal)
            elif accrual < 0.05: score += 1   # FIX: was 0.03/0.08 — thresholds tightened
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


def compute_earnings_revision_score(row: pd.Series) -> float:
    """
    Earnings Revision Score (0–5):
    Combines two signals:
    1. Revision breadth  — ratio of up-revisions vs down-revisions (30-day)
    2. Revision magnitude — % change in consensus EPS over 90 days
    3. Recent acceleration — 7-day ratio as a bonus for very recent momentum

    Research basis: Zacks (1979), Chan/Jegadeesh/Lakonishok (1996) —
    earnings revisions are one of the strongest short-term alpha factors.
    """
    score = 0.0
    try:
        # Signal 1: Revision breadth (30-day) — core signal
        ratio_30d = _safe(row.get("rev_ratio_30d"))
        if not np.isnan(ratio_30d):
            if ratio_30d > 0.6:    score += 2.0   # strong net upgrades
            elif ratio_30d > 0.2:  score += 1.5   # moderate net upgrades
            elif ratio_30d > 0:    score += 1.0   # slight net upgrades
            elif ratio_30d > -0.2: score += 0.5   # roughly balanced
            # ratio_30d <= -0.2: no points (net downgrades)

        # Signal 2: EPS estimate change over 90 days — magnitude
        eps_chg_90 = _safe(row.get("eps_revision_pct_90d"))
        if not np.isnan(eps_chg_90):
            if eps_chg_90 > 0.05:    score += 1.5   # estimates up >5%
            elif eps_chg_90 > 0.02:  score += 1.0   # estimates up >2%
            elif eps_chg_90 > 0:     score += 0.5   # any positive revision
            elif eps_chg_90 < -0.05: score -= 0.5   # significant downgrade penalty

        # Signal 3: Recent acceleration (7-day ratio) — bonus
        ratio_7d = _safe(row.get("rev_ratio_7d"))
        if not np.isnan(ratio_7d) and ratio_7d > 0.5:
            score += 0.5   # very recent upgrade momentum

    except Exception:
        pass
    return float(np.clip(score, 0, 5))
