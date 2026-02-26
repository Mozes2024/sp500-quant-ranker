"""
Unit tests for S&P 500 Ranking System v5.3
Tests critical scoring functions to prevent silent regressions.
Run: python -m pytest test_scoring.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
import unittest.mock

# Add parent dir to path
sys.path.insert(0, os.path.dirname(__file__))

# Mock subprocess.check_call to skip pip install during import
with unittest.mock.patch('subprocess.check_call'):
    from sp500_github import (
        compute_piotroski,
        compute_altman,
        compute_roic,
        compute_fcf_metrics,
        compute_earnings_quality,
        compute_valuation_score,
        compute_composite,
        build_pillar_scores,
        _safe,
        _coverage,
        PILLAR_MAP,
        CFG,
    )


# ═══════════════════════════════════════════════════
#  HELPER: Build a mock stock row
# ═══════════════════════════════════════════════════

def make_row(**kwargs):
    """Create a pd.Series simulating a stock's data row."""
    defaults = {
        "sector": "Information Technology",
        "totalAssets": 100_000_000_000,
        "returnOnAssets": 0.12,
        "returnOnEquity": 0.25,
        "operatingCashflow": 15_000_000_000,
        "earningsGrowth": 0.15,
        "debtToEquity": 80,
        "currentRatio": 1.8,
        "payoutRatio": 0.3,
        "sharesOutstanding": 1_000_000_000,
        "marketCap": 200_000_000_000,
        "currentPrice": 200.0,
        "grossMargins": 0.45,
        "totalRevenue": 80_000_000_000,
        "revenueGrowth": 0.10,
        "totalStockholdersEquity": 50_000_000_000,
        "ebitda": 20_000_000_000,
        "totalDebt": 30_000_000_000,
        "totalCash": 10_000_000_000,
        "freeCashflow": 12_000_000_000,
        "netIncomeToCommon": 10_000_000_000,
        "profitMargins": 0.125,
        "trailingPE": 20,
        "forwardPE": 18,
        "pegRatio": 1.3,
        "priceToSalesTrailing12Months": 2.5,
        "priceToBook": 4.0,
        "enterpriseToEbitda": 11.0,
        "workingCapital": 20_000_000_000,
        "currentAssets": 40_000_000_000,
        "currentLiabilities": 20_000_000_000,
        "beta": 1.1,
        "recommendationMean": 2.0,
        "shortRatio": 3.0,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


# ═══════════════════════════════════════════════════
#  TEST: _safe utility
# ═══════════════════════════════════════════════════

class TestSafe:
    def test_normal_value(self):
        assert _safe(42.0) == 42.0

    def test_none_returns_default(self):
        assert np.isnan(_safe(None))
        assert _safe(None, 0) == 0

    def test_nan_returns_default(self):
        assert np.isnan(_safe(float("nan")))
        assert _safe(float("nan"), -1) == -1

    def test_string_returns_default(self):
        assert np.isnan(_safe("not_a_number"))


# ═══════════════════════════════════════════════════
#  TEST: Piotroski F-Score
# ═══════════════════════════════════════════════════

class TestPiotroski:
    def test_healthy_company_scores_high(self):
        row = make_row()
        score = compute_piotroski(row)
        assert score >= 6, f"Healthy company should score ≥6, got {score}"

    def test_distressed_company_scores_low(self):
        row = make_row(
            returnOnAssets=-0.05,
            operatingCashflow=-5_000_000_000,
            earningsGrowth=-0.20,
            debtToEquity=300,
            currentRatio=0.5,
            grossMargins=0.10,
            revenueGrowth=-0.15,
        )
        score = compute_piotroski(row)
        assert score <= 3, f"Distressed company should score ≤3, got {score}"

    def test_score_range(self):
        row = make_row()
        score = compute_piotroski(row)
        assert 0 <= score <= 9, f"Score must be 0-9, got {score}"

    def test_missing_data_doesnt_crash(self):
        row = pd.Series({"sector": "Energy"})
        score = compute_piotroski(row)
        assert 0 <= score <= 9


# ═══════════════════════════════════════════════════
#  TEST: Altman Z-Score
# ═══════════════════════════════════════════════════

class TestAltman:
    def test_healthy_tech_company(self):
        row = make_row(sector="Information Technology")
        z = compute_altman(row)
        assert z > 2.6, f"Healthy tech company should be safe zone (>2.6), got {z}"

    def test_manufacturing_uses_z_model(self):
        row = make_row(sector="Industrials")
        z = compute_altman(row)
        assert z is not None and not np.isnan(z)

    def test_financials_returns_nan(self):
        row = make_row(sector="Financials")
        z = compute_altman(row)
        assert np.isnan(z), "Financials should return NaN (model not applicable)"

    def test_real_estate_returns_nan(self):
        row = make_row(sector="Real Estate")
        z = compute_altman(row)
        assert np.isnan(z), "Real Estate should return NaN"

    def test_missing_data_returns_nan(self):
        row = pd.Series({"sector": "Energy"})
        z = compute_altman(row)
        assert z is None or np.isnan(z)


# ═══════════════════════════════════════════════════
#  TEST: ROIC
# ═══════════════════════════════════════════════════

class TestROIC:
    def test_positive_roic(self):
        row = make_row()
        roic = compute_roic(row)
        assert roic > 0, f"Healthy company should have positive ROIC, got {roic}"

    def test_formula_correctness(self):
        row = make_row(
            ebitda=10_000_000_000,
            totalStockholdersEquity=30_000_000_000,
            totalDebt=20_000_000_000,
            totalCash=5_000_000_000,
        )
        roic = compute_roic(row)
        # NOPAT = ebitda * 0.82 * (1 - 0.21)
        nopat = 10e9 * 0.82 * 0.79
        inv_cap = 30e9 + 20e9 - 5e9
        expected = nopat / inv_cap
        assert abs(roic - expected) < 0.001, f"ROIC should be {expected:.4f}, got {roic:.4f}"


# ═══════════════════════════════════════════════════
#  TEST: FCF Metrics
# ═══════════════════════════════════════════════════

class TestFCF:
    def test_fcf_yield_positive(self):
        row = make_row()
        metrics = compute_fcf_metrics(row)
        assert metrics["fcf_yield"] > 0

    def test_fcf_margin_positive(self):
        row = make_row()
        metrics = compute_fcf_metrics(row)
        assert metrics["fcf_margin"] > 0

    def test_negative_fcf(self):
        row = make_row(freeCashflow=-5_000_000_000)
        metrics = compute_fcf_metrics(row)
        assert metrics["fcf_yield"] < 0


# ═══════════════════════════════════════════════════
#  TEST: Earnings Quality
# ═══════════════════════════════════════════════════

class TestEarningsQuality:
    def test_high_quality_company(self):
        row = make_row(
            freeCashflow=15_000_000_000,
            netIncomeToCommon=10_000_000_000,
            roic=0.25,
            profitMargins=0.15,
            grossMargins=0.50,
        )
        score = compute_earnings_quality(row)
        assert score >= 3, f"High quality company should score ≥3, got {score}"

    def test_score_range(self):
        row = make_row()
        score = compute_earnings_quality(row)
        assert 0 <= score <= 5, f"Score must be 0-5, got {score}"


# ═══════════════════════════════════════════════════
#  TEST: Composite Score
# ═══════════════════════════════════════════════════

class TestComposite:
    def test_composite_in_range(self):
        row = pd.Series({col: 60.0 for col in PILLAR_MAP.values()})
        score = compute_composite(row)
        assert 10 <= score <= 100, f"Composite should be 10-100, got {score}"

    def test_all_nan_returns_nan(self):
        row = pd.Series({col: np.nan for col in PILLAR_MAP.values()})
        score = compute_composite(row)
        assert np.isnan(score)

    def test_partial_data(self):
        row = pd.Series({col: np.nan for col in PILLAR_MAP.values()})
        row["pillar_valuation"] = 80
        row["pillar_profitability"] = 70
        score = compute_composite(row)
        assert not np.isnan(score), "Should compute with partial data"


# ═══════════════════════════════════════════════════
#  TEST: Weight Integrity
# ═══════════════════════════════════════════════════

class TestWeights:
    def test_weights_sum_to_one(self):
        total = sum(CFG["weights"].values())
        assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"

    def test_all_pillars_have_weights(self):
        for key in PILLAR_MAP:
            assert key in CFG["weights"], f"Missing weight for pillar '{key}'"

    def test_no_negative_weights(self):
        for key, w in CFG["weights"].items():
            assert w >= 0, f"Negative weight for '{key}': {w}"


# ═══════════════════════════════════════════════════
#  TEST: Double Counting Prevention (v5.3 fix)
# ═══════════════════════════════════════════════════

class TestDoubleCounting:
    """Verify that the v5.3 fixes actually removed double counting."""

    def test_fcf_to_ni_not_in_growth_pillar(self):
        """fcf_to_ni should only appear in FCF pillar, not growth."""
        # Build a DataFrame with known values
        data = {
            "ticker": ["TEST"],
            "sector": ["Information Technology"],
            "revenueGrowth": [0.15],
            "earningsGrowth": [0.20],
            "eps_revision_pct_30d": [0.05],  # NEW: replaces fcf_to_ni in growth
            "tr_asset_growth": [0.10],
            "earnings_revision_score": [3.0],
            "fcf_to_ni": [1.5],              # should NOT affect growth pillar
            "fcf_yield": [0.05],
            "fcf_margin": [0.15],
        }
        df = pd.DataFrame(data)

        # The growth pillar should use s_eps_rev_30d, not s_fcf_ni_g
        # We verify by checking column names in the function
        # (structural test — the code doesn't reference fcf_to_ni in growth)
        import inspect
        source = inspect.getsource(build_pillar_scores)

        # Growth section should NOT contain s_fcf_ni_g
        growth_section = source[source.index("# 3. Growth"):source.index("# 4. Earnings")]
        assert "s_fcf_ni_g" not in growth_section, "fcf_to_ni still in growth pillar!"
        assert "s_eps_rev_30d" in growth_section, "eps_revision_pct_30d missing from growth pillar"


# ═══════════════════════════════════════════════════
#  TEST: Coverage calculation
# ═══════════════════════════════════════════════════

class TestCoverage:
    def test_full_coverage(self):
        from sp500_github import CORE_METRIC_COLS
        row = pd.Series({c: 1.0 for c in CORE_METRIC_COLS})
        cov = _coverage(row, CORE_METRIC_COLS)
        assert cov == 1.0

    def test_zero_coverage(self):
        from sp500_github import CORE_METRIC_COLS
        row = pd.Series({c: np.nan for c in CORE_METRIC_COLS})
        cov = _coverage(row, CORE_METRIC_COLS)
        assert cov == 0.0

    def test_partial_coverage(self):
        from sp500_github import CORE_METRIC_COLS
        row = pd.Series({c: np.nan for c in CORE_METRIC_COLS})
        for c in CORE_METRIC_COLS[:10]:
            row[c] = 1.0
        cov = _coverage(row, CORE_METRIC_COLS)
        assert 0.4 < cov < 0.7


# ═══════════════════════════════════════════════════
#  TEST: JSON output structure (smoke test)
# ═══════════════════════════════════════════════════

class TestJSONOutput:
    def test_json_file_is_valid(self):
        """Smoke test: verify sp500_data.json is valid and has expected fields."""
        import json
        json_path = os.path.join(os.path.dirname(__file__), "sp500_data.json")
        if not os.path.exists(json_path):
            pytest.skip("sp500_data.json not found (run pipeline first)")

        with open(json_path) as f:
            data = json.load(f)

        assert "generated" in data
        assert "count" in data
        assert "data" in data
        assert len(data["data"]) > 400, f"Expected >400 stocks, got {len(data['data'])}"

        # Check first record has required fields
        first = data["data"][0]
        required = ["rank", "ticker", "company", "sector", "composite", "valuation"]
        for field in required:
            assert field in first, f"Missing field '{field}' in JSON record"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
