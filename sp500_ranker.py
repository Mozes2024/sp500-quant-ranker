# ============================================================
# S&P 500 ADVANCED RANKING SYSTEM v5.2 – GitHub Edition
# Headless + GitHub Actions + Persistent Cache + Dynamic Valuation
# ============================================================

import subprocess, sys, os, pickle, time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Install dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install",
    "yfinance", "pandas", "numpy", "openpyxl==3.1.2", "requests",
    "beautifulsoup4", "matplotlib", "seaborn", "tqdm", "scipy", "-q"])

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
from bs4 import BeautifulSoup
from tqdm import tqdm
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.2f}".format)

# ====================== CONFIG ======================
CFG = {
    "weights": {
        "valuation": 0.19, "profitability": 0.16, "growth": 0.13,
        "earnings_quality": 0.08, "fcf_quality": 0.13,
        "financial_health": 0.09, "momentum": 0.09,
        "analyst": 0.11, "piotroski": 0.02,
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
os.makedirs("artifacts", exist_ok=True)

# ====================== TIPRANKS ======================
# (אותו קוד כמו בגרסה שלך – לא שיניתי)

TR_URL = "https://mobile.tipranks.com/api/stocks/stockAnalysisOverview"
TR_HEADERS = { ... }  # העתק את כל ה-TR_HEADERS, _CONSENSUS, _TREND, _SENTIMENT, _SMA, _parse_tipranks, fetch_tipranks מהגרסה שלך (הכל נשאר זהה)

# ====================== כל שאר הפונקציות ======================
# העתק כאן את כל הקוד מהגרסה v5.1 שלך החל מ-get_sp500_tickers() ועד _print_summary (כולל build_sector_thresholds, compute_valuation_score, load_cache/save_cache וכו')

# ====================== HEADLESS MAIN ======================
if __name__ == "__main__":
    print("🚀 Starting S&P 500 Ranking v5.2 – GitHub Actions")
    df = run_pipeline(use_cache=True)   # ← הריצה האוטומטית
    print("✅ Done! Artifacts saved to /artifacts")
