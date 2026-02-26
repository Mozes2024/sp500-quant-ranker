# S&P 500 Advanced AI Ranking System v5.3

### @MOZES — Fixed Edition

---

## v5.3 Changelog (Critical Fixes)

### 🔴 Logic Fixes (Double Counting)
- **Removed `fcf_to_ni` from Growth Pillar** — was double-counted with FCF Pillar. Replaced with `eps_revision_pct_30d`
- **Simplified Analyst Pillar** — SmartScore already aggregates consensus, insider, hedge, news. New: SS 60%, PT 25%, Yahoo 15%
- **Piotroski weight → 0%** — proxy-based F-Score without Y/Y data = noise
- **Analyst pillar weight → 0%** — double-count risk with SmartScore internals

### 🟠 New Signals
- **Short Ratio** — added to Momentum Pillar. Already in Yahoo data, was unused
- **Insider $ as % Market Cap** — normalized insider buy/sell vs binary direction
- **`eps_revision_pct_30d`** — short-term EPS revision velocity

### 🟡 UI/UX Improvements
- **Compare Mode** — select 2-4 stocks, side-by-side with best-value highlighting
- **Watchlist** — localStorage-based star system with dedicated tab
- **Export CSV** — downloads filtered table for current view
- **Light/Dark Theme** — toggle with persistence
- **Stale Data Warning** — banner when data >48h old

### 🔴 Infrastructure
- **TipRanks Fallback Weights** — auto redistribution when TR API fails
- **31 Unit Tests** — Piotroski, Altman, ROIC, FCF, EQ, Composite, Coverage, Weights, Double Counting
- **`_TR_AVAILABLE` flag** — tracked globally, in JSON for frontend

### Running Tests
```bash
python -m pytest test_scoring.py -v
```
