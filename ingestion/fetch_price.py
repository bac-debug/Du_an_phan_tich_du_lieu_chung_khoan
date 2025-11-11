import argparse
import yfinance as yf
import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)




def fetch_price(ticker: str, start: str, end: str):
df = yf.download(ticker + '.VN', start=start, end=end, progress=False)
if df.empty:
# fallback to ticker alone
df = yf.download(ticker, start=start, end=end, progress=False)
df.to_csv(DATA_DIR / f"{ticker}_prices.csv")
print(f"Saved prices to {DATA_DIR / f'{ticker}_prices.csv'}")
return df




if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--ticker', required=True)
parser.add_argument('--start', default='2020-01-01')
parser.add_argument('--end', default='2025-09-01')
args = parser.parse_args()
fetch_price(args.ticker, args.start, args.end)