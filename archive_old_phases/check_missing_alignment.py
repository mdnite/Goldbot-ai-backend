import yfinance as yf
from fredapi import Fred
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

START = "2010-01-01"

gold = yf.download("GC=F", start=START, auto_adjust=True)
gold.columns = gold.columns.droplevel(1)
gold = gold[["Close"]].rename(columns={"Close": "gold"})

dxy = yf.download("DX-Y.NYB", start=START, auto_adjust=True)
dxy.columns = dxy.columns.droplevel(1)
dxy = dxy[["Close"]].rename(columns={"Close": "dxy"})

real_yield = fred.get_series("DFII10", observation_start=START).rename("real_yield")
fed_rate = fred.get_series("DFF", observation_start=START).rename("fed_rate")

print("gold rows:", len(gold), "| range:", gold.index.min().date(), "->", gold.index.max().date())
print("dxy rows:", len(dxy), "| range:", dxy.index.min().date(), "->", dxy.index.max().date())
print("real_yield rows:", len(real_yield), "| range:", real_yield.index.min().date(), "->", real_yield.index.max().date())
print("fed_rate rows:", len(fed_rate), "| range:", fed_rate.index.min().date(), "->", fed_rate.index.max().date())

all_dates = gold.index.union(dxy.index).union(real_yield.index).union(fed_rate.index)
print("\nTong so ngay xuat hien it nhat 1 indicator (union):", len(all_dates))

inner = pd.concat([gold["gold"], dxy["dxy"], real_yield, fed_rate], axis=1, join="inner")
print("So ngay con lai sau INNER JOIN (du ca 4):", len(inner))

dropped = len(all_dates) - len(inner)
print(f"So ngay bi loai bo neu dung inner join: {dropped} ({dropped/len(all_dates)*100:.2f}% tren tong so ngay xuat hien it nhat 1 indicator)")

for name, idx in [("gold", gold.index), ("dxy", dxy.index), ("real_yield", real_yield.index), ("fed_rate", fed_rate.index)]:
    missing = all_dates.difference(idx)
    print(f"{name}: thieu {len(missing)} / {len(all_dates)} ngay so voi union ({len(missing)/len(all_dates)*100:.2f}%)")