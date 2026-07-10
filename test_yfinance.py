import yfinance as yf

gold = yf.download("GC=F", start="2010-01-01", auto_adjust=True)
print("=== GC=F (giá vàng) ===")
print("Số dòng:", gold.shape[0])
print(gold[["Close"]].head(3))
print(gold[["Close"]].tail(3))
print("Số dòng thiếu (NaN):", gold["Close"].isna().sum())

print()

dxy = yf.download("DX-Y.NYB", start="2010-01-01", auto_adjust=True)
print("=== DX-Y.NYB (DXY) ===")
print("Số dòng:", dxy.shape[0])
print(dxy[["Close"]].head(3))
print(dxy[["Close"]].tail(3))
print("Số dòng thiếu (NaN):", dxy["Close"].isna().sum())