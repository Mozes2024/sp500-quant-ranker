# ranker/utils.py — Shared utility functions
import numpy as np
import pandas as pd

def _safe(val, default=np.nan):
    """Safely convert value to float, returning default for None/NaN/non-numeric."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except Exception:
        return default

# ════════════════════════════════════════════════════════════
#  COMPUTED METRICS
# ════════════════════════════════════════════════════════════

# _safe() defined above (before Yahoo Finance section) — FIX B4


def _coverage(row: pd.Series, cols: list) -> float:
    return sum(1 for c in cols if not pd.isna(row.get(c, np.nan))) / max(len(cols), 1)
