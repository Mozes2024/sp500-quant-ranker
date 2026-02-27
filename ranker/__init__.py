# ranker/__init__.py — S&P 500 Quant Ranker v5.3 modular package
#
# This package provides a clean modular structure for the ranking system.
# Import anything directly: `from ranker import compute_roic, CFG`
#
# Module layout:
#   config.py        — CFG, weights, thresholds, constants
#   utils.py         — _safe(), _coverage() shared helpers
#   data_yahoo.py    — Yahoo Finance fetching (parallel)
#   data_tipranks.py — TipRanks fetching
#   data_tickers.py  — S&P 500 list (Wikipedia + fallbacks)
#   metrics.py       — Piotroski, Altman Z, ROIC, FCF, EQ, Revisions
#   pillars.py       — Pillar score building + sector percentile
#   composite.py     — Composite scoring, valuation score, coverage
#   export_excel.py  — Excel export + formatting
#   export_json.py   — JSON export for dashboard
#   charts.py        — matplotlib/seaborn visualizations
#   breakout.py      — Breakout scanner signal merge
#   cache.py         — Cache load/save
#   summary.py       — Console summary output
#   pipeline.py      — Main orchestration (run_pipeline)

__version__ = "5.3"
