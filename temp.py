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
print("Ma trận tương quan (GIÁ THÔ - chỉ để tham khảo, KHÔNG dùng để kết luận):")
print(df.corr())

# Giá thô (level) có xu hướng dài hạn -> tương quan Pearson trên level dễ bị "giả"
# (spurious correlation) do cùng trend, không phản ánh quan hệ ngày-qua-ngày thật.
# Gold/DXY là GIÁ -> dùng % thay đổi (pct_change). real_yield/fed_rate là LÃI SUẤT
# (có thể gần 0 hoặc âm) -> dùng hiệu số tuyệt đối (diff), không dùng %.
returns = pd.DataFrame({
    "gold": df["gold"].pct_change(),
    "dxy": df["dxy"].pct_change(),
    "real_yield": df["real_yield"].diff(),
    "fed_rate": df["fed_rate"].diff(),
}).dropna()

print()
print("Ma trận tương quan (% THAY ĐỔI / DIFF - dùng cái này để kết luận):")
print(returns.corr())