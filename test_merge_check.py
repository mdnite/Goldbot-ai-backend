import yfinance as yf
from fredapi import Fred
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

# Gold
gold = yf.download("GC=F", start="2010-01-01", auto_adjust=True)
gold.columns = gold.columns.droplevel(1)          # bỏ tầng "Ticker"
gold = gold[["Close"]].rename(columns={"Close": "gold"})

# DXY
dxy = yf.download("DX-Y.NYB", start="2010-01-01", auto_adjust=True)
dxy.columns = dxy.columns.droplevel(1)
dxy = dxy[["Close"]].rename(columns={"Close": "dxy"})

# FRED (không lỗi MultiIndex, trả Series thẳng)
real_yield = fred.get_series("DFII10", observation_start="2010-01-01").rename("real_yield")
fed_rate = fred.get_series("DFF", observation_start="2010-01-01").rename("fed_rate")

# Gộp inner join giữ ngày có đủ cả 4 giá trị
df = pd.concat([gold["gold"], dxy["dxy"], real_yield, fed_rate], axis=1, join="inner")
df = df.sort_index()

print("Số dòng sau khi gộp:", len(df))
print(df.tail(5))
print()
print("Ma trận tương quan:")
print(df.corr())