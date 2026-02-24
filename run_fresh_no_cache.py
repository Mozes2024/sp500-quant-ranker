"""Run the ranking pipeline with no cache so RS vs Market and all 10 pillars are computed and exported."""
import os
import sys

CACHE_FILE = "sp500_cache_v10.pkl"
if os.path.exists(CACHE_FILE):
    os.remove(CACHE_FILE)
    print(f"  Removed {CACHE_FILE} so this run does a full rebuild.")
else:
    print("  No cache file found; full rebuild will run.")

from sp500_github import run_pipeline
run_pipeline(use_cache=False)
sys.exit(0)
