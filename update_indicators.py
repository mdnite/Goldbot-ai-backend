import sqlite3
from datetime import datetime

import yfinance as yf
from fredapi import Fred
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

DB_PATH = "indicators.db"
START = "2010-01-01"

# Gold
gold = yf.download("GC=F", start=START, auto_adjust=True)
gold.columns = gold.columns.droplevel(1)
gold = gold[["Close"]].rename(columns={"Close": "gold"})

# DXY
dxy = yf.download("DX-Y.NYB", start=START, auto_adjust=True)
dxy.columns = dxy.columns.droplevel(1)
dxy = dxy[["Close"]].rename(columns={"Close": "dxy"})

# FRED
real_yield = fred.get_series("DFII10", observation_start=START).rename("real_yield")
fed_rate = fred.get_series("DFF", observation_start=START).rename("fed_rate")

# Gộp inner join - chỉ giữ ngày có đủ cả 4 giá trị, không lưu thiếu
df = pd.concat([gold["gold"], dxy["dxy"], real_yield, fed_rate], axis=1, join="inner")
df = df.sort_index()

# Reshape sang EAV (long format): date, indicator, value
long_df = df.reset_index(names="date").melt(id_vars="date", var_name="indicator", value_name="value")
long_df["date"] = long_df["date"].dt.strftime("%Y-%m-%d")

fetched_at = datetime.now().isoformat()
long_df["fetched_at"] = fetched_at

conn = sqlite3.connect(DB_PATH)
conn.execute("""
    CREATE TABLE IF NOT EXISTS indicators (
        date TEXT,
        indicator TEXT,
        value REAL,
        fetched_at TEXT,
        PRIMARY KEY (date, indicator)
    )
""")
conn.executemany(
    "INSERT OR REPLACE INTO indicators (date, indicator, value, fetched_at) VALUES (?, ?, ?, ?)",
    long_df[["date", "indicator", "value", "fetched_at"]].itertuples(index=False, name=None),
)
conn.commit()
conn.close()

print(f"Da ghi {len(long_df)} dong vao {DB_PATH}")
print(f"Khoang ngay: {df.index.min().date()} -> {df.index.max().date()} ({len(df)} ngay x 4 indicator)")
print(f"fetched_at: {fetched_at}")
