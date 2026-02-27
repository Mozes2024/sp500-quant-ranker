# ranker/data_tickers.py — S&P 500 ticker list fetching
import pandas as pd
import requests
from bs4 import BeautifulSoup
from ranker.config import WIKI_HEADERS

def get_sp500_tickers() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    try:
        resp = requests.get(url, headers=WIKI_HEADERS, timeout=15)
        resp.raise_for_status()
        soup  = BeautifulSoup(resp.text, "html.parser")
        table = (soup.find("table", {"id": "constituents"})
                 or soup.find("table", {"class": "wikitable"}))
        if table is None:
            raise ValueError("Table not found")
        rows = []
        for tr in table.find_all("tr")[1:]:
            cols = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cols) >= 4:
                rows.append({"ticker":   cols[0].replace(".", "-"),
                             "name":     cols[1],
                             "sector":   cols[2],
                             "industry": cols[3]})
        if rows:
            df = pd.DataFrame(rows)
            print(f"✅  {len(df)} tickers (BeautifulSoup)")
            return df
    except Exception as e:
        print(f"  ⚠️  Strategy 1 failed: {e}")

    try:
        tables = pd.read_html(url, attrs={"id": "constituents"}) or pd.read_html(url)
        raw = tables[0]
        raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
        tc = next((c for c in raw.columns if "symbol" in c or "ticker" in c), raw.columns[0])
        nc = next((c for c in raw.columns if "security" in c or "name"   in c), raw.columns[1])
        sc = next((c for c in raw.columns if "sector"  in c),                    raw.columns[2])
        ic = next((c for c in raw.columns if "industry" in c),                   raw.columns[3])
        df = pd.DataFrame({
            "ticker":   raw[tc].astype(str).str.replace(".", "-", regex=False),
            "name":     raw[nc].astype(str),
            "sector":   raw[sc].astype(str),
            "industry": raw[ic].astype(str),
        })
        print(f"✅  {len(df)} tickers (read_html)")
        return df
    except Exception as e:
        print(f"  ⚠️  Strategy 2 failed: {e}")

    try:
        raw = pd.read_csv(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
            "main/data/constituents.csv")
        raw.columns = [c.lower() for c in raw.columns]
        df = pd.DataFrame({
            "ticker":   raw["symbol"].str.replace(".", "-", regex=False),
            "name":     raw.get("name",     raw.get("security",     "")),
            "sector":   raw.get("sector",   "Unknown"),
            "industry": raw.get("sub-industry", raw.get("industry", "Unknown")),
        })
        print(f"✅  {len(df)} tickers (GitHub CSV)")
        return df
    except Exception as e:
        raise RuntimeError(f"All 3 ticker strategies failed: {e}")
