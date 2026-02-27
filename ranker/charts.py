# ranker/charts.py — Matplotlib/Seaborn visualizations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from ranker.config import PILLAR_MAP

def plot_all(df: pd.DataFrame):
    sns.set_style("whitegrid")

    # Top 30 composite
    fig, ax = plt.subplots(figsize=(14, 8))
    top30  = df.nlargest(30, "composite_score")
    colors = plt.cm.RdYlGn(top30["composite_score"] / 100)
    bars   = ax.barh(top30["ticker"][::-1], top30["composite_score"][::-1],
                     color=colors[::-1], edgecolor="white")
    ax.set_xlim(0, 108)
    ax.set_xlabel("Composite Score", fontsize=12)
    ax.set_title("Top 30 S&P 500 – Composite Score v5.3", fontsize=14, fontweight="bold")
    for bar, s in zip(bars, top30["composite_score"][::-1]):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{s:.1f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig("artifacts/top30_composite.png", dpi=150)
    plt.close()

    # Sector medians
    fig, ax = plt.subplots(figsize=(13, 5))
    sec = df.groupby("sector")["composite_score"].median().sort_values(ascending=False)
    sec.plot(kind="bar", ax=ax,
             color=plt.cm.coolwarm_r(np.linspace(0, 1, len(sec))), edgecolor="white")
    ax.set_title("Median Composite Score by Sector", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=40)
    plt.tight_layout()
    plt.savefig("artifacts/sector_scores.png", dpi=150)
    plt.close()

    # Scatter cheap vs quality
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(df["valuation_score"], df["composite_score"],
                    c=df["composite_score"], cmap="RdYlGn", alpha=0.65, s=35)
    plt.colorbar(sc, label="Composite Score")
    quad = df[(df["valuation_score"] > 65) & (df["composite_score"] > 65)]
    for _, r in quad.iterrows():
        ax.annotate(r["ticker"], (r["valuation_score"], r["composite_score"]),
                    fontsize=7, alpha=0.85)
    ax.axvline(50, color="grey", ls="--", alpha=0.4)
    ax.axhline(df["composite_score"].median(), color="grey", ls="--", alpha=0.4)
    ax.set_xlabel("Valuation Score (100=Cheap)")
    ax.set_ylabel("Composite Score")
    ax.set_title("Quality vs Valuation Quadrant Map", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("artifacts/valuation_vs_quality.png", dpi=150)
    plt.close()

    # SmartScore distribution
    if "tr_smart_score" in df.columns and df["tr_smart_score"].notna().sum() > 10:
        fig, ax = plt.subplots(figsize=(10, 5))
        for sv in sorted(df["tr_smart_score"].dropna().unique()):
            ax.bar(sv, (df["tr_smart_score"] == sv).sum(),
                   color=plt.cm.RdYlGn(sv / 10), edgecolor="white", width=0.8)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("SmartScore")
        ax.set_ylabel("# Stocks")
        ax.set_title("TipRanks SmartScore Distribution", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig("artifacts/smartscore_dist.png", dpi=150)
        plt.close()

    # Multi-timeframe momentum top 20
    if "perf_12m" in df.columns:
        top20 = df.nlargest(20, "composite_score")
        fig, ax = plt.subplots(figsize=(14, 6))
        x, w = np.arange(len(top20)), 0.2
        for i, (col, label, color) in enumerate([
            ("perf_12m","12M","#1976D2"), ("perf_6m","6M","#42A5F5"),
            ("perf_3m","3M","#81D4FA"),  ("perf_1m","1M","#B3E5FC"),
        ]):
            ax.bar(x + i*w, top20[col].fillna(0)*100, w,
                   label=label, color=color, edgecolor="white")
        ax.set_xticks(x + 1.5*w)
        ax.set_xticklabels(top20["ticker"], rotation=45, fontsize=8)
        ax.set_ylabel("Return %")
        ax.legend()
        ax.set_title("Multi-Timeframe Momentum – Top 20", fontsize=13, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.8)
        plt.tight_layout()
        plt.savefig("artifacts/momentum_decomp.png", dpi=150)
        plt.close()

    # Radar top 5
    _plot_radar(df.nlargest(5, "composite_score"))


def _plot_radar(df_top: pd.DataFrame):
    pillar_cols = list(PILLAR_MAP.values())
    labels  = [c.replace("pillar_","").replace("_","\n").title() for c in pillar_cols]
    N       = len(pillar_cols)
    angles  = [n / N * 2 * np.pi for n in range(N)] + [0]
    colors  = ["#2196F3","#4CAF50","#FF9800","#E91E63","#9C27B0"]

    fig, axes = plt.subplots(1, min(5, len(df_top)), figsize=(20, 4),
                             subplot_kw=dict(polar=True))
    if len(df_top) == 1:
        axes = [axes]

    for i, (_, row) in enumerate(df_top.iterrows()):
        ax   = axes[i]
        vals = [_safe(row.get(c), 50) for c in pillar_cols] + [_safe(row.get(pillar_cols[0]), 50)]
        ax.plot(angles, vals, color=colors[i], linewidth=2)
        ax.fill(angles, vals, color=colors[i], alpha=0.22)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=6.5)
        ax.set_ylim(0, 100)
        ss     = row.get("tr_smart_score", np.nan)
        ss_str = f"  SS:{ss:.0f}" if not pd.isna(ss) else ""
        ax.set_title(f"{row['ticker']}\n{row['composite_score']:.1f}{ss_str}",
                     size=9, fontweight="bold", pad=10)

    fig.suptitle("Pillar Breakdown – Top 5 (v5.3)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("artifacts/top5_radar.png", dpi=150)
    plt.close()
