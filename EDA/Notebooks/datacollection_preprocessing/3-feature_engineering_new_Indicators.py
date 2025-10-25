import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from ta.momentum import RSIIndicator
from scipy.stats import skew, kurtosis

def calculate_final_technical_indicators(file_path):
    try:
        df = pd.read_csv(file_path, parse_dates=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {file_path}")

    required_cols = ['Ticker', 'Close', 'High', 'Low', 'Open', 'Volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.sort_values(by=['Ticker', 'Date'] if 'Date' in df.columns else ['Ticker'], inplace=True)
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    returns = df['returns']
    risk_free_rate = 0.02 / 252

    for window in [10, 50, 200]:
        df[f'EMA_{window}'] = df.groupby('Ticker')['Close'].transform(lambda x: EMAIndicator(close=x, window=window).ema_indicator())

    macd_data = df.groupby('Ticker').apply(
        lambda x: pd.DataFrame({
            'MACD': MACD(close=x['Close']).macd(),
            'MACD_Signal': MACD(close=x['Close']).macd_signal()
        }, index=x.index)
    )
    df[['MACD', 'MACD_Signal']] = macd_data.reset_index(level=0, drop=True)[['MACD', 'MACD_Signal']]

    df['RSI_14'] = df.groupby('Ticker')['Close'].transform(lambda x: RSIIndicator(close=x, window=14).rsi())

    df['ATR_14'] = df.groupby('Ticker').apply(
        lambda x: AverageTrueRange(high=x['High'], low=x['Low'], close=x['Close'], window=14).average_true_range()
    ).reset_index(level=0, drop=True)

    df['Volatility_21'] = df.groupby('Ticker')['returns'].transform(lambda x: x.rolling(21).std() * np.sqrt(252))

    df['OBV'] = df.groupby('Ticker').apply(
        lambda x: OnBalanceVolumeIndicator(close=x['Close'], volume=x['Volume']).on_balance_volume()
    ).reset_index(level=0, drop=True)

    df['Avg_Daily_Volume'] = df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(252).mean())

    cumulative = df.groupby('Ticker')['returns'].transform(lambda x: (1 + x).cumprod())
    peak = cumulative.groupby(df['Ticker']).transform('cummax')
    drawdown = (cumulative - peak) / peak
    df['Max_Drawdown_21'] = drawdown.groupby(df['Ticker']).transform(lambda x: x.rolling(21).min())
    df['Calmar_Ratio_21'] = df.groupby('Ticker')['returns'].transform(lambda x: x.rolling(21).mean()) / abs(df['Max_Drawdown_21'])
    df['Cumulative_Return'] = cumulative - 1
    df['CAGR'] = df.groupby('Ticker')['Close'].transform(lambda x: (x / x.shift(252)) ** (1 / 1) - 1)

    df['Skewness'] = df.groupby('Ticker')['returns'].transform(lambda x: x.rolling(252).apply(skew, raw=True))
    df['Kurtosis'] = df.groupby('Ticker')['returns'].transform(lambda x: x.rolling(252).apply(kurtosis, raw=True))

    benchmark = df.groupby('Date')['Close'].transform('mean') if 'Date' in df.columns else df.groupby('Ticker')['Close'].transform('mean')
    benchmark_returns = benchmark.pct_change()
    df['Beta'] = df.groupby('Ticker')['returns'].transform(lambda x: x.rolling(21).cov(benchmark_returns) / benchmark_returns.rolling(21).var())
    df['Alpha'] = df.groupby('Ticker')['returns'].transform(lambda x: x.rolling(21).mean()) - (
        risk_free_rate + df['Beta'] * (benchmark_returns.rolling(21).mean() - risk_free_rate)
    )

    df['Up_Capture_Ratio'] = df.groupby('Ticker')['returns'].transform(lambda x: x.rolling(252).mean()) / benchmark_returns.rolling(252).mean()

    adx_grouped = df.groupby('Ticker').apply(
        lambda x: pd.DataFrame({
            'ADX_14': ADXIndicator(high=x['High'], low=x['Low'], close=x['Close'], window=14).adx(),
            'DMI_plus_14': ADXIndicator(high=x['High'], low=x['Low'], close=x['Close'], window=14).adx_pos(),
            'DMI_minus_14': ADXIndicator(high=x['High'], low=x['Low'], close=x['Close'], window=14).adx_neg(),
        }, index=x.index)
    )
    df[['ADX_14', 'DMI_plus_14', 'DMI_minus_14']] = adx_grouped.reset_index(level=0, drop=True)

    return df


if __name__ == "__main__":
    file_path = "/Users/imperator/Documents/STUDIES/UNIVERSITY OF GHANA/RESEARCH WORK/CORCHIL KELLY KWAME/PORTFOLIO OPTIMIZATION/PROJECT CODE/Dimensionality-Reduction-PortfolioOptimization/EDA/ETFs_datasets/Final_csv/1-combined_etfs.csv"
    output_path = "/Users/imperator/Documents/STUDIES/UNIVERSITY OF GHANA/RESEARCH WORK/CORCHIL KELLY KWAME/PORTFOLIO OPTIMIZATION/Project Code/Dimensionality-Reduction-PortfolioOptimization/EDA/ETFs_datasets/Final_csv/technical_indicators.csv"

    try:
        indicators_df = calculate_final_technical_indicators(file_path)
        print(indicators_df.head())
        indicators_df.to_csv(output_path, index=False)
        print(f"Indicators saved to {output_path}")
    except Exception as e:
        print(f"Error: {e}")
