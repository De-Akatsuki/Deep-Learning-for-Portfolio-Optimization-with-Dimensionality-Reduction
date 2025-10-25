import yfinance as yf
import pandas as pd
from pathlib import Path

# Define the directory to save raw data
DATA_DIR = Path('/Users/imperator/Documents/STUDIES/UNIVERSITY OF GHANA/RESEARCH WORK/CORCHIL KELLY KWAME/PORTFOLIO OPTIMIZATION/PROJECT CODE/Dimensionality-Reduction-PortfolioOptimization/EDA/ETFs_datasets/RAW/ETFS')
DATA_DIR.mkdir(parents=True, exist_ok=True)

# List of ETF tickers (example: top-performing ETFs)
etfs = ["VTI","AGG","DBC", "VIX"]

# Function to fetch historical data for ETFs
def fetch_historical_data(tickers: list, period: str = "10y"):
    for ticker in tickers:
        try:
            print(f"Fetching historical data for {ticker}...")
            
            # Download historical data using yfinance
            df = yf.download(ticker, period=period)
            
            # Skip if no data is returned
            if df.empty:
                print(f"No data found for {ticker}. Skipping...")
                continue
            
            # Save raw data to CSV file in the raw_data folder
            file_path = DATA_DIR / f"{ticker}_historical_data_new.csv"
            df.to_csv(file_path)
            
            print(f"Data saved for {ticker} at {file_path}")
        
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

# Fetch historical data for the list of ETFs
fetch_historical_data(etfs)
