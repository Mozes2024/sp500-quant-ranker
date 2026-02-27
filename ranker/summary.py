# ranker/summary.py — Console summary output
import pandas as pd
from ranker.config import PILLAR_MAP

def _print_summary(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("  TOP 20 STOCKS")
    print("=" * 65)
    show = ["rank","ticker","name","sector","composite_score",
            "valuation_score","tr_smart_score","tr_analyst_consensus",
            "earnings_quality_score","earnings_revision_score",
            "piotroski_score","altman_z","coverage"]
    print(df[[c for c in show if c in df.columns]].head(20).to_string(index=False))

    print("\n  SECTOR MEDIAN COMPOSITE SCORES")
    print("-" * 45)
    print(df.groupby("sector")["composite_score"].median()
            .sort_values(ascending=False).round(1).to_string())

    if "tr_smart_score" in df.columns and df["tr_smart_score"].notna().any():
        top_ss = df[df["tr_smart_score"] >= 8][
            ["rank","ticker","composite_score","tr_smart_score",
             "tr_analyst_consensus","sector"]]
        if not top_ss.empty:
            print(f"\n  TIPRANKS SMARTSCORE >= 8  ({len(top_ss)} stocks)")
            print("-" * 55)
            print(top_ss.to_string(index=False))
