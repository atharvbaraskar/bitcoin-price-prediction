import yfinance as yf

btc = yf.download(
    "BTC-USD",
    start="2015-01-01",
    end="2026-09-22",
    interval="1d"
)

print(btc.head())