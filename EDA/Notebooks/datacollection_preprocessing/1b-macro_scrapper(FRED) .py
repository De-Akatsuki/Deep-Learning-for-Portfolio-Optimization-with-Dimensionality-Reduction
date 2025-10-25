import pandas as pd
import ssl
from fredapi import Fred
import os
from datetime import datetime
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

# from config import config

# # Use the FRED API key
# FRED_API_KEY = config.fred_api_key

from fredapi import Fred

FRED_API_KEY = "dbad3eb85c21ada837fbb5e0f4c87025"
fred = Fred(api_key=FRED_API_KEY)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FredDataFetcher:
    def __init__(self, api_key):
        self.fred = Fred(api_key=api_key)
        self.setup_ssl()
        
    @staticmethod
    def setup_ssl():
        # Create default SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    
    def fetch_macro_data(self, features_dict):
        macro_data = pd.DataFrame()
        
        for name, code in features_dict.items():
            try:
                logger.info(f"Fetching {name} data...")
                series = self.fred.get_series(code)
                macro_data[name] = series
            except Exception as e:
                logger.error(f"Error fetching {name}: {e}")
                
        return macro_data

def main():
    # FRED API key
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        raise ValueError("FRED API key not found in environment variables")
    
    macro_features = {
    # Inflation
    'cpi': 'CPIAUCSL',  # Consumer Price Index
    'core_cpi': 'CPILFESL',  # Core CPI (excludes food/energy)
    'pce_price_index': 'PCEPI',  # Personal Consumption Expenditures Index

    # Growth & Output
    'real_gdp': 'GDPC1',  # Real Gross Domestic Product
    'industrial_production': 'INDPRO',  # Industrial Production Index
    'durable_goods_orders': 'DGORDER',  # Durable Goods New Orders

    # Labor Market
    'unemployment_rate': 'UNRATE',  # Unemployment Rate
    'nonfarm_payrolls': 'PAYEMS',  # Nonfarm Payroll Employment
    'jobless_claims': 'ICSA',  # Initial Unemployment Claims

    # Monetary Policy & Credit
    'fed_funds_rate': 'FEDFUNDS',  # Federal Funds Rate
    'treasury_10y': 'GS10',  # 10-Year Treasury Constant Maturity Rate
    'treasury_3m': 'GS3M',  # 3-Month Treasury Bill Rate
    'baa_yield': 'BAA10Y',  # Moody's BAA Corporate Bond Yield
    'aaa_yield': 'AAA10Y',  # Moody's AAA Corporate Bond Yield
    'consumer_credit': 'TOTALSL',  # Total Consumer Credit

    # Housing & Real Estate
    'mortgage_rate_30y': 'MORTGAGE30US',  # 30-Year Fixed Rate Mortgage Average
    'housing_starts': 'HOUST',  # Housing Starts
    'building_permits': 'PERMIT',  # Building Permits
    'new_home_sales': 'HSN1F',  # New One-Family Houses Sold

    # Sentiment
    'umich_consumer_sentiment': 'UMCSENT',  # University of Michigan Consumer Sentiment Index
    }

    
    try:
        # Initialize fetcher
        fetcher = FredDataFetcher(api_key)
        
        # Fetch data
        macro_data = fetcher.fetch_macro_data(macro_features)
        
        # Save to CSV
        output_file = f"macro_data_{datetime.now().strftime('%Y%m%d')}.csv"
        macro_data.to_csv(output_file)
        logger.info(f"Data saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()