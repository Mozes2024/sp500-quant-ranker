# ranker/breakout.py — Breakout scanner signal merge
import json, os
import numpy as np
import pandas as pd

def merge_breakout_signals(df: pd.DataFrame) -> pd.DataFrame:
    import json as _json, os as _os
    bp = "breakout_signals.json"
    if not _os.path.exists(bp):
        print("  ⚠  breakout_signals.json not found — skipping breakout merge")
        for col in ["breakout_score","breakout_rank","breakout_phase","breakout_rr",
                    "breakout_rs","breakout_stop","has_vcp","vcp_quality",
                    "breakout_entry_quality","breakout_reasons"]:
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
    print(f"  🔀 Overlap with S&P 500: {n_overlap} stocks in both systems")
    return df
