# ranker/cache.py — Cache load/save
import os, pickle
from datetime import datetime, timedelta
import pandas as pd
from ranker.config import CFG, CACHE_FILE

_SECTOR_THRESHOLDS = {}

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
        rs_ok = "rs_12m" in data.columns and not data["rs_12m"].isna().all()
        pr_ok = "pillar_relative_strength" in data.columns and not data["pillar_relative_strength"].isna().all()
        if not rs_ok or not pr_ok:
            print(f"  ℹ️  Cache missing or empty RS/pillar_relative_strength — full rebuild")
            return None
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
