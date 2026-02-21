name: MOZES Automated Stock Scanner
on:
  schedule:
    - cron: '15 9 * * 1-5'
  workflow_dispatch:

permissions:
  contents: write

jobs:
  screen-stocks:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          token: ${{ github.token }}

      - name: Get current date
        id: date
        run: echo "date=$(date +'%Y-%m-%d')" >> $GITHUB_OUTPUT

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run screening
        run: |
          mkdir -p data/daily_scans
          mkdir -p data/logs
          python run_optimized_scan.py \
            --conservative \
            --git-storage \
            --min-market-cap 2000000000 \
            --min-rel-volume 1.5

      - name: Commit and Push results
        if: success()
        run: |
          git config --local user.email "github-actions[bot]@users.noreply.github.com"
          git config --local user.name "github-actions[bot]"

          DAY_OF_WEEK=$(date +'%A')

          TIMESTAMP_FILE=$(ls data/daily_scans/optimized_scan_*.txt 2>/dev/null | head -n 1)

          if [ -n "$TIMESTAMP_FILE" ] && [ -f "$TIMESTAMP_FILE" ]; then
            mv "$TIMESTAMP_FILE" "data/daily_scans/scan_${DAY_OF_WEEK}.txt"
            cp "data/daily_scans/scan_${DAY_OF_WEEK}.txt" "data/daily_scans/latest_optimized_scan.txt"
            echo "✅ Saved as scan_${DAY_OF_WEEK}.txt and latest_optimized_scan.txt"
          else
            # נסה למצוא כל קובץ סקאן קיים וליצור latest ממנו
            EXISTING=$(ls data/daily_scans/*.txt 2>/dev/null | head -n 1)
            if [ -n "$EXISTING" ]; then
              cp "$EXISTING" "data/daily_scans/latest_optimized_scan.txt"
              echo "✅ Created latest_optimized_scan.txt from $EXISTING"
            else
              echo "⚠ No scan file found anywhere"
              ls data/daily_scans/ || echo "Directory empty"
            fi
          fi

          # הוספה מפורשת של כל קבצי הסקאן כולל latest
          git add -f data/daily_scans/ 2>/dev/null || true
          git add data/logs/*.log 2>/dev/null || true

          if git diff --staged --quiet; then
            echo "No changes to commit"
          else
            git commit -m "Automated Scan Results: ${{ steps.date.outputs.date }} ($DAY_OF_WEEK)"
            git push
          fi

      - name: Upload screening results as artifact
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: sp500-ranking-results-${{ steps.date.outputs.date }}
          path: |
            data/daily_scans/
            data/logs/
          if-no-files-found: warn
          retention-days: 90

      # ══════════════════════════════════════════════════════
      # שלבים חדשים: המרה ל-JSON ודחיפה לראנקר
      # ══════════════════════════════════════════════════════
      - name: Convert scan to JSON
        if: success()
        run: |
          python3 << 'PYEOF'
          import re, json, glob
          from datetime import datetime, date
          from pathlib import Path

          # חיפוש קובץ סקאן לפי סדר עדיפויות
          txt = None
          candidates = ["data/daily_scans/latest_optimized_scan.txt"]
          candidates.append(f"data/daily_scans/scan_{date.today().strftime('%A')}.txt")
          candidates += sorted(glob.glob("data/daily_scans/optimized_scan_*.txt"), reverse=True)
          candidates += sorted(glob.glob("data/daily_scans/*.txt"), reverse=True)

          for c in candidates:
              p = Path(c)
              if p.exists() and p.stat().st_size > 1000:
                  txt = p
                  print(f"📄 Using: {txt}")
                  break

          if txt is None:
              print("⚠ No scan file found — skipping")
              import os; os.makedirs("data/daily_scans", exist_ok=True)
              files = list(Path("data/daily_scans").glob("*"))
              print(f"Files in directory: {files}")
              exit(0)

          content = txt.read_text(encoding="utf-8")

          def find(pattern, text, flags=0):
              m = re.search(pattern, text, flags)
              return m.group(1).strip() if m else None

          blocks = re.split(r"(?=⭐ BUY #\d+:)", content)
          signals = []
          for block in blocks[1:]:
              m = re.search(r"⭐ BUY #(\d+): (\w+) \| Score: ([\d.]+)/110", block)
              if not m: continue
              rank, ticker, score = int(m.group(1)), m.group(2), float(m.group(3))
              phase      = re.search(r"^Phase: (\d)", block, re.M)
              entry_qual = re.search(r"Entry Quality: (\w+)", block)
              stop_loss  = re.search(r"Stop Loss: \$([\d.]+)", block)
              rr         = re.search(r"Risk/Reward: ([\d.]+):1 \(Risk \$([\d.]+), Reward \$([\d.]+)\)", block)
              rs         = re.search(r"\bRS: ([\d.]+)", block)
              bkout      = re.search(r"Breakout: \$([\d.]+)", block)
              vcp        = re.search(r"VCP: (.+?)(?:\n)", block)
              vcp_q      = re.search(r"quality: (\d+)/100", block)
              reasons_raw = re.findall(r"  • (.+)", block)
              reasons = [re.sub(r"^[🟢🟡🔴✓✗⭐]\s*", "", r).strip() for r in reasons_raw[:5]]
              signals.append({
                  "rank": rank, "ticker": ticker, "breakout_score": score,
                  "phase": int(phase.group(1)) if phase else None,
                  "entry_quality": entry_qual.group(1) if entry_qual else None,
                  "stop_loss": float(stop_loss.group(1)) if stop_loss else None,
                  "risk": float(rr.group(2)) if rr else None,
                  "reward": float(rr.group(3)) if rr else None,
                  "risk_reward": float(rr.group(1)) if rr else None,
                  "rs": float(rs.group(1)) if rs else None,
                  "breakout_price": float(bkout.group(1)) if bkout else None,
                  "has_vcp": bool(vcp),
                  "vcp_quality": int(vcp_q.group(1)) if vcp_q else None,
                  "vcp_desc": vcp.group(1).strip() if vcp else None,
                  "reasons": reasons,
              })

          payload = {
              "scan_date":         find(r"Scan Date: (.+)", content) or "",
              "generated":         find(r"Generated: (.+)", content) or "",
              "converted_at":      datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
              "market_regime":     find(r"Market Regime: (.+)", content) or "",
              "total_buy_signals": int(find(r"Buy Signals: (\d+)", content) or 0),
              "phase2_pct":        float(find(r"Phase 2.*?(\d+\.\d+)%", content, re.DOTALL) or 0),
              "top_signals_count": len(signals),
              "top_signals":       signals,
          }

          with open("breakout_signals.json", "w", encoding="utf-8") as f:
              json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

          print(f"✅ Converted {len(signals)} signals → breakout_signals.json")
          print(f"   Regime: {payload['market_regime']}")
          PYEOF

      - name: Push breakout_signals.json to Ranker repo
        if: success()
        run: |
          if [ ! -f "breakout_signals.json" ]; then
            echo "⚠ breakout_signals.json not found — skipping"
            exit 0
          fi

          CONTENT=$(base64 -w 0 breakout_signals.json)

          RESPONSE=$(curl -s \
            -H "Authorization: token ${{ secrets.RANKER_REPO_TOKEN }}" \
            -H "Accept: application/vnd.github.v3+json" \
            "https://api.github.com/repos/Mozes2024/SP500-Quant-Ranker_2026/contents/breakout_signals.json")

          SHA=$(echo "$RESPONSE" | python3 -c \
            "import sys,json; d=json.load(sys.stdin); print(d.get('sha',''))" 2>/dev/null || echo "")

          DATE_STR=$(date +'%Y-%m-%d')
          if [ -n "$SHA" ]; then
            PAYLOAD="{\"message\":\"📈 Update breakout signals ${DATE_STR}\",\"content\":\"${CONTENT}\",\"sha\":\"${SHA}\"}"
          else
            PAYLOAD="{\"message\":\"📈 Add breakout signals ${DATE_STR}\",\"content\":\"${CONTENT}\"}"
          fi

          RESULT=$(curl -s -X PUT \
            -H "Authorization: token ${{ secrets.RANKER_REPO_TOKEN }}" \
            -H "Accept: application/vnd.github.v3+json" \
            "https://api.github.com/repos/Mozes2024/SP500-Quant-Ranker_2026/contents/breakout_signals.json" \
            -d "$PAYLOAD")

          echo "$RESULT" | python3 -c "
          import sys,json
          d=json.load(sys.stdin)
          if 'content' in d:
              print('✅ breakout_signals.json pushed to SP500-Quant-Ranker_2026!')
          else:
              print('❌ Error:', d.get('message','unknown'))
              exit(1)
          "
